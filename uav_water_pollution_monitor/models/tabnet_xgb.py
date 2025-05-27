import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetClassifier
import xgboost as xgb
import numpy as np
import joblib # For saving/loading XGBoost model
import os

from utils.config_parser import ConfigParser
from utils.logger import setup_logger

logger = setup_logger("tabnet_xgb_ensemble")
config = ConfigParser()

class TabNetXGBEnsemble(nn.Module):
    def __init__(self, input_dim, output_dim, tabnet_params=None, xgb_params=None, device=None):
        super(TabNetXGBEnsemble, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"TabNetXGB Ensemble using device: {self.device}")

        self.input_dim = input_dim
        self.output_dim = output_dim # Number of classes

        # TabNet Classifier
        default_tabnet_params = {
            "n_d": 8, "n_a": 8, "n_steps": 3, "gamma": 1.3,
            "n_independent": 2, "n_shared": 2, "lambda_sparse": 1e-3,
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=2e-2),
            "scheduler_params": {"step_size":10, "gamma":0.9},
            "scheduler_fn": torch.optim.lr_scheduler.StepLR,
            "mask_type": 'sparsemax', # Paper uses 'sparsemax' or 'entmax'
            "verbose": 10, # Print progress every 10 epochs
            "device_name": self.device
        }
        _tabnet_params_config = config.get("tabnet_params", {})
        # Smart merge: config values override defaults
        merged_tabnet_params = {**default_tabnet_params, **_tabnet_params_config}
        if tabnet_params: # Explicit params to constructor take highest precedence
             merged_tabnet_params.update(tabnet_params)

        # Ensure optimizer_fn and scheduler_fn are actual functions if specified as strings in config
        if isinstance(merged_tabnet_params.get("optimizer_fn"), str):
            opt_fn_str = merged_tabnet_params["optimizer_fn"].split('.')
            opt_module = torch.optim if opt_fn_str[0] == "torch" else __import__(opt_fn_str[0])
            for comp in opt_fn_str[1:]:
                opt_module = getattr(opt_module, comp)
            merged_tabnet_params["optimizer_fn"] = opt_module

        if isinstance(merged_tabnet_params.get("scheduler_fn"), str):
            sch_fn_str = merged_tabnet_params["scheduler_fn"].split('.')
            sch_module = torch.optim.lr_scheduler if sch_fn_str[0] == "torch" else __import__(sch_fn_str[0])
            for comp in sch_fn_str[1:]:
                sch_module = getattr(sch_module, comp)
            merged_tabnet_params["scheduler_fn"] = sch_module


        logger.info(f"Initializing TabNet with params: {merged_tabnet_params}")
        self.tabnet = TabNetClassifier(**merged_tabnet_params)
        # Note: TabNetClassifier is designed for training with its .fit() method.
        # For ensemble, we might use its feature transformer part or train it first, then use its output.
        # The paper says: "The combined feature set T serves as input to a modified TabNet architecture"
        # "The output of the final step N is then passed through a ReLU activation function,
        # batch normalization, and a fully connected layer: H = ReLU(FN), HBN = BatchNorm(H), O = FC(HBN)"
        # "Finally, an XGBoost classifier is employed for the ultimate prediction task: y_hat = XGBoost(O)"
        # This implies TabNet is used as a feature processor, and its output O is fed to XGBoost.
        # TabNetClassifier's `predict_proba` gives class probabilities. We need the intermediate features `O`.
        # This might require accessing internal TabNet components or modifying TabNetClassifier.
        # A simpler approach: train TabNet for classification, then use its penultimate layer's output.
        # Or, use TabNetRegressor if the task is to predict intermediate features (less common).

        # For now, we'll train TabNet, then XGBoost on TabNet's PROBABILITIES or extracted features.
        # The paper's Fig 1 shows TabNet output (after FC layer) feeding XGBoost.
        # This "FC Layer" (O in the paper) is likely the layer before softmax in TabNet.

        # XGBoost Classifier
        default_xgb_params = {
            'objective': 'multi:softprob', # Output probabilities for each class
            'eval_metric': 'mlogloss',
            'eta': 0.1,
            'max_depth': 6,
            'num_class': self.output_dim, # Crucial for multi-class
            'use_label_encoder': False # Suppress warning for newer XGBoost
        }
        _xgb_params_config = config.get("xgb_params", {})
        merged_xgb_params = {**default_xgb_params, **_xgb_params_config}
        if xgb_params:
            merged_xgb_params.update(xgb_params)

        logger.info(f"Initializing XGBoost with params: {merged_xgb_params}")
        self.xgb_params = merged_xgb_params
        self.xgboost = xgb.XGBClassifier(**self.xgb_params)

        self.tabnet_fitted = False
        self.xgboost_fitted = False

    def fit_tabnet(self, X_train, y_train, X_val=None, y_val=None, tabnet_fit_params=None):
        """
        Fits the TabNet part of the ensemble.
        X_train, X_val should be NumPy arrays.
        """
        logger.info("Fitting TabNet model...")
        default_fit_params = {
            "max_epochs": config.get("epochs_ensemble_tabnet", 50), # Separate epochs for TabNet part
            "patience": 10,
            "batch_size": config.get("batch_size_ensemble_tabnet", 1024), # TabNet often uses larger batches
            "virtual_batch_size": config.get("virtual_batch_size_ensemble_tabnet", 128),
            "num_workers": 0,
            "drop_last": False
        }
        _fit_params_config = config.get("tabnet_fit_params", {})
        merged_fit_params = {**default_fit_params, **_fit_params_config}
        if tabnet_fit_params:
            merged_fit_params.update(tabnet_fit_params)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            logger.info(f"TabNet fitting with eval set of shape: {X_val.shape}")
        else:
            logger.info("TabNet fitting without a separate validation set for early stopping within TabNet fit.")

        self.tabnet.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=['mlogloss' if self.output_dim > 1 else 'auc'], # mlogloss for multi-class, auc for binary
            **merged_fit_params
        )
        self.tabnet_fitted = True
        logger.info("TabNet model fitting complete.")

    def get_tabnet_features(self, X):
        """
        Extracts features from TabNet. This is what the paper refers to as 'O'.
        This needs to access an intermediate representation from the TabNet model.
        TabNetClassifier's `tab_network_` is the actual nn.Module.
        The `tab_network_.forward` returns (output, M_loss). `output` is before final activation for classification.
        """
        if not self.tabnet_fitted:
            raise RuntimeError("TabNet model must be fitted before extracting features.")
        
        self.tabnet.network.eval() # Set to evaluation mode
        # Convert X to tensor if it's a numpy array
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float().to(self.tabnet.device)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.to(self.tabnet.device)
        else:
            raise ValueError("Input X must be a NumPy array or PyTorch tensor.")

        with torch.no_grad():
            # The `tab_network_.forward` returns (processed_features, m_loss)
            # The `processed_features` are the logits before the final GSoftmax/Softmax application
            # This should correspond to 'O' in the paper if FC layer is the last layer of TabNet.
            tabnet_output_features, _ = self.tabnet.network(X_tensor)
        
        return tabnet_output_features.cpu().numpy()


    def fit_xgboost(self, X_train_tabnet_features, y_train, X_val_tabnet_features=None, y_val=None, xgb_fit_params=None):
        """
        Fits the XGBoost part of the ensemble using features extracted from TabNet.
        """
        if not self.tabnet_fitted:
            logger.warning("Fitting XGBoost, but TabNet was not explicitly fitted. Ensure features are from a trained TabNet.")

        logger.info("Fitting XGBoost model...")
        default_fit_params = {
            "early_stopping_rounds": 10,
            "verbose": True
        }
        merged_fit_params = {**default_fit_params, **(xgb_fit_params or {})}

        eval_set_xgb = []
        if X_val_tabnet_features is not None and y_val is not None:
            eval_set_xgb = [(X_val_tabnet_features, y_val)]
            logger.info(f"XGBoost fitting with eval set of shape: {X_val_tabnet_features.shape}")
        else:
            logger.info("XGBoost fitting without a separate validation set for early stopping.")
        
        self.xgboost.fit(
            X_train_tabnet_features, y_train,
            eval_set=eval_set_xgb,
            **merged_fit_params
        )
        self.xgboost_fitted = True
        logger.info("XGBoost model fitting complete.")

    def predict_proba(self, X_combined_features):
        if not self.tabnet_fitted or not self.xgboost_fitted:
            raise RuntimeError("Both TabNet and XGBoost models must be fitted before prediction.")
        
        tabnet_output_features = self.get_tabnet_features(X_combined_features)
        xgb_proba = self.xgboost.predict_proba(tabnet_output_features)
        return xgb_proba

    def predict(self, X_combined_features):
        probas = self.predict_proba(X_combined_features)
        return np.argmax(probas, axis=1)

    def save_model(self, directory):
        os.makedirs(directory, exist_ok=True)
        # Save TabNet model (pytorch_tabnet saves as a zip file)
        tabnet_model_path = os.path.join(directory, "tabnet_model") # No .zip, library handles it
        self.tabnet.save_model(tabnet_model_path) # Returns path with .zip
        logger.info(f"TabNet model saved to {tabnet_model_path}.zip")

        # Save XGBoost model
        xgb_model_path = os.path.join(directory, "xgboost_model.json") # or .ubj for binary
        self.xgboost.save_model(xgb_model_path)
        # For full python object: joblib.dump(self.xgboost, xgb_model_path_joblib)
        logger.info(f"XGBoost model saved to {xgb_model_path}")

    def load_model(self, directory):
        # Load TabNet model
        tabnet_model_zip_path = os.path.join(directory, "tabnet_model.zip")
        if not os.path.exists(tabnet_model_zip_path):
            logger.error(f"TabNet model file not found at {tabnet_model_zip_path}")
            raise FileNotFoundError(f"TabNet model file not found at {tabnet_model_zip_path}")
        self.tabnet.load_model(tabnet_model_zip_path)
        self.tabnet_fitted = True
        logger.info(f"TabNet model loaded from {tabnet_model_zip_path}")

        # Load XGBoost model
        xgb_model_path = os.path.join(directory, "xgboost_model.json")
        if not os.path.exists(xgb_model_path):
            logger.error(f"XGBoost model file not found at {xgb_model_path}")
            raise FileNotFoundError(f"XGBoost model file not found at {xgb_model_path}")
        self.xgboost.load_model(xgb_model_path)
        self.xgboost_fitted = True
        logger.info(f"XGBoost model loaded from {xgb_model_path}")


if __name__ == '__main__':
    # Dummy data for testing
    # Combined features: ViT_features + YOLO_bbox_features + Sensor_features
    # Example: ViT (768) + YOLO (4 for bbox) + Sensor (2 for pH, TDS) = 774 features
    input_dim_test = 50 # Simplified for testing
    output_dim_test = 3  # 3 classes
    num_samples = 200
    num_val_samples = 50

    X_train_np = np.random.rand(num_samples, input_dim_test).astype(np.float32)
    y_train_np = np.random.randint(0, output_dim_test, num_samples)
    X_val_np = np.random.rand(num_val_samples, input_dim_test).astype(np.float32)
    y_val_np = np.random.randint(0, output_dim_test, num_val_samples)

    ensemble_model = TabNetXGBEnsemble(input_dim=input_dim_test, output_dim=output_dim_test)

    # Fit TabNet
    logger.info("--- Fitting TabNet ---")
    # TabNet fit params can be fine-tuned, e.g., max_epochs, patience for early stopping
    ensemble_model.fit_tabnet(X_train_np, y_train_np, X_val_np, y_val_np, tabnet_fit_params={"max_epochs": 5}) # Few epochs for test

    # Extract features from TabNet for XGBoost training
    logger.info("--- Extracting TabNet features ---")
    X_train_tabnet_feats = ensemble_model.get_tabnet_features(X_train_np)
    X_val_tabnet_feats = ensemble_model.get_tabnet_features(X_val_np)
    logger.info(f"Shape of TabNet features for training XGBoost: {X_train_tabnet_feats.shape}")

    # Fit XGBoost
    logger.info("--- Fitting XGBoost ---")
    ensemble_model.fit_xgboost(X_train_tabnet_feats, y_train_np, X_val_tabnet_feats, y_val_np, xgb_fit_params={"n_estimators": 20}) # Few estimators for test

    # Predict
    logger.info("--- Making Predictions ---")
    sample_test_data = np.random.rand(10, input_dim_test).astype(np.float32)
    predictions_proba = ensemble_model.predict_proba(sample_test_data)
    predictions_class = ensemble_model.predict(sample_test_data)
    logger.info(f"Sample prediction probabilities:\n{predictions_proba}")
    logger.info(f"Sample predicted classes:\n{predictions_class}")

    # Save and Load model
    model_save_dir = "temp_ensemble_model"
    logger.info(f"--- Saving Model to {model_save_dir} ---")
    ensemble_model.save_model(model_save_dir)

    logger.info(f"--- Loading Model from {model_save_dir} ---")
    loaded_ensemble_model = TabNetXGBEnsemble(input_dim=input_dim_test, output_dim=output_dim_test)
    loaded_ensemble_model.load_model(model_save_dir)
    
    logger.info("--- Making Predictions with Loaded Model ---")
    loaded_predictions_class = loaded_ensemble_model.predict(sample_test_data)
    logger.info(f"Loaded model predicted classes:\n{loaded_predictions_class}")
    assert np.array_equal(predictions_class, loaded_predictions_class), "Predictions from original and loaded model do not match!"
    logger.info("Predictions match. Save/Load test successful.")

    # Clean up
    # import shutil
    # if os.path.exists(model_save_dir):
    #     shutil.rmtree(model_save_dir)