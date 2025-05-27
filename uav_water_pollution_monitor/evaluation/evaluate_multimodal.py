import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib # For loading label encoder and scaler
from tqdm import tqdm

from utils.config_parser import ConfigParser
from utils.logger import setup_logger
from models.vit_extractor import ViTExtractor
from models.sensor_feature_loader import SensorFeatureLoader
from models.tabnet_xgb import TabNetXGBEnsemble
from utils.metrics import get_classification_metrics
from utils.visualize_results import plot_confusion_matrix
from models.train_ensemble import load_image_paths_and_labels, prepare_features_for_split # Re-use helper functions

logger = setup_logger("multimodal_evaluator")
config = ConfigParser()

# Paths and Parameters
PROCESSED_DATA_DIR = config.get("processed_data_dir")
IMAGES_BASE_DIR = os.path.join(PROCESSED_DATA_DIR, "yolo_dataset", "images") # Test split here
IMAGE_TO_CLASS_LABEL_FILE = os.path.join(PROCESSED_DATA_DIR, "image_class_labels.csv") # Must contain test image labels
YOLO_BBOX_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, "yolo_bbox_features.csv") # For test images

ENSEMBLE_MODEL_LOAD_PATH = os.path.join(config.get("models_dir"), "tabnet_xgb_ensemble_v1")
RESULTS_DIR = config.get("results_dir", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    logger.info("Starting Multi-modal Ensemble Model Evaluation")

    if not os.path.exists(ENSEMBLE_MODEL_LOAD_PATH):
        logger.error(f"Trained ensemble model directory not found at: {ENSEMBLE_MODEL_LOAD_PATH}")
        logger.error("Please train the ensemble model first using 'models/train_ensemble.py'.")
        return

    # --- 0. Initialize Feature Extractors ---
    vit_ext = ViTExtractor()
    sensor_load = SensorFeatureLoader()
    
    yolo_bbox_df = None
    if os.path.exists(YOLO_BBOX_FEATURES_FILE):
        yolo_bbox_df = pd.read_csv(YOLO_BBOX_FEATURES_FILE)
        logger.info(f"Loaded YOLO BBox features from {YOLO_BBOX_FEATURES_FILE} for evaluation.")
    else:
        logger.warning(f"YOLO BBox features file not found: {YOLO_BBOX_FEATURES_FILE}. YOLO features will be NaNs/zeros for eval.")


    # --- 1. Load Test Data (Image Paths and True Labels) ---
    test_image_paths, test_labels_str = load_image_paths_and_labels(split='test')
    if not test_image_paths:
        logger.error("No test image paths loaded. Ensure 'image_class_labels.csv' has 'test/' entries and images exist.")
        return

    # --- 2. Load Label Encoder and Encode Test Labels ---
    label_encoder_path = os.path.join(ENSEMBLE_MODEL_LOAD_PATH, "label_encoder.pkl")
    if not os.path.exists(label_encoder_path):
        logger.error(f"Label encoder not found at {label_encoder_path}. Cannot proceed with evaluation.")
        return
    label_encoder = joblib.load(label_encoder_path)
    
    try:
        y_test_true = label_encoder.transform(test_labels_str)
    except ValueError as e:
        logger.error(f"Error transforming test labels: {e}. Some labels in test set might be unseen during training.")
        logger.error(f"Known classes by encoder: {label_encoder.classes_}")
        logger.error(f"Unique labels in current test set: {np.unique(test_labels_str)}")
        # Option: filter out unseen labels or map them to an 'unknown' class if your model handles it.
        # For now, we'll error out.
        return

    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_.tolist()
    logger.info(f"Loaded label encoder. Test set has {len(y_test_true)} samples. Num classes: {num_classes}")

    # --- 3. Prepare Combined Features for Test Set ---
    # Option to load pre-extracted features if saved during training or a separate feature extraction step
    X_test_combined_path = os.path.join(config.get("processed_data_dir", "data/processed"), "combined_features", "X_test_combined.npy")
    force_reextract_test = config.get("evaluation.force_feature_reextraction_test", True) # Default to re-extract for eval

    if not force_reextract_test and os.path.exists(X_test_combined_path):
        logger.info("Loading pre-extracted combined features for test set...")
        X_test_combined = np.load(X_test_combined_path)
        # y_test_loaded = np.load(os.path.join(os.path.dirname(X_test_combined_path), "y_test.npy"))
        # if not np.array_equal(y_test_true, y_test_loaded): # Sanity check
        #     logger.warning("Loaded y_test does not match current y_test. Re-extracting features.")
        #     force_reextract_test = True
    
    if force_reextract_test or not os.path.exists(X_test_combined_path):
        logger.info("Extracting combined features for test set...")
        X_test_combined = prepare_features_for_split(test_image_paths, vit_ext, sensor_load, yolo_bbox_df, "test")
        
        # Save for future use if needed (optional for eval script)
        # np.save(X_test_combined_path, X_test_combined)
        # np.save(os.path.join(os.path.dirname(X_test_combined_path), "y_test.npy"), y_test_true)
        # logger.info("Saved extracted combined features for test set.")

    if X_test_combined.shape[0] == 0:
        logger.error("No combined features available for the test set. Aborting.")
        return
    if X_test_combined.shape[0] != len(y_test_true):
        logger.error(f"Mismatch between number of test feature samples ({X_test_combined.shape[0]}) and labels ({len(y_test_true)}). Aborting.")
        return
    
    X_test_final = X_test_combined.astype(np.float32)
    logger.info(f"Test features prepared. Shape: {X_test_final.shape}")

    # --- 4. Load Trained Ensemble Model ---
    # The input_dim should match what the model was trained with.
    # We can infer it from X_test_final.shape[1]
    input_dim_eval = X_test_final.shape[1]
    ensemble_model = TabNetXGBEnsemble(input_dim=input_dim_eval, output_dim=num_classes)
    try:
        ensemble_model.load_model(ENSEMBLE_MODEL_LOAD_PATH)
        logger.info(f"Ensemble model loaded successfully from: {ENSEMBLE_MODEL_LOAD_PATH}")
    except Exception as e:
        logger.error(f"Failed to load ensemble model: {e}")
        return

    # --- 5. Make Predictions on Test Set ---
    logger.info("Making predictions on the test set...")
    y_test_pred_proba = ensemble_model.predict_proba(X_test_final)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)

    # --- 6. Calculate and Log Metrics ---
    test_metrics_weighted = get_classification_metrics(y_test_true, y_test_pred, class_names=class_names, average='weighted')
    logger.info(f"Multimodal Test Metrics (Weighted):")
    logger.info(f"  Accuracy: {test_metrics_weighted['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics_weighted['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics_weighted['recall']:.4f}")
    logger.info(f"  F1-Score: {test_metrics_weighted['f1_score']:.4f}")

    test_metrics_per_class = get_classification_metrics(y_test_true, y_test_pred, class_names=class_names, average=None)
    logger.info(f"Multimodal Test Metrics (Per-Class):")
    results_df_data = []
    for i, name in enumerate(class_names):
        prec = test_metrics_per_class[f'precision_{name}']
        rec = test_metrics_per_class[f'recall_{name}']
        f1 = test_metrics_per_class[f'f1_score_{name}']
        logger.info(f"  Class: {name} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
        results_df_data.append({"class": name, "precision": prec, "recall": rec, "f1_score": f1})
    
    # Save per-class metrics to CSV
    results_csv_path = os.path.join(RESULTS_DIR, "multimodal_test_metrics_per_class.csv")
    pd.DataFrame(results_df_data).to_csv(results_csv_path, index=False)
    logger.info(f"Per-class metrics saved to {results_csv_path}")

    # Save overall weighted metrics
    overall_metrics_path = os.path.join(RESULTS_DIR, "multimodal_test_metrics_overall.csv")
    overall_dict = {k: v for k, v in test_metrics_weighted.items() if k != 'confusion_matrix'}
    pd.DataFrame([overall_dict]).to_csv(overall_metrics_path, index=False)
    logger.info(f"Overall weighted metrics saved to {overall_metrics_path}")


    # Plot confusion matrix for the test set
    cm_test_path = os.path.join(RESULTS_DIR, "multimodal_test_confusion_matrix.png")
    plot_confusion_matrix(test_metrics_weighted['confusion_matrix'], class_names, title="Multimodal Ensemble Test Confusion Matrix", output_path=cm_test_path)

    logger.info("Multimodal evaluation complete.")

if __name__ == "__main__":
    # Before running:
    # 1. Ensure the ensemble model is trained and saved in ENSEMBLE_MODEL_LOAD_PATH.
    # 2. Ensure `image_class_labels.csv` contains entries for the 'test' split.
    # 3. Ensure test images exist in `data/processed/yolo_dataset/images/test/`.
    # 4. (Optional) `yolo_bbox_features.csv` should contain features for test images.
    # 5. (Optional) `sensor_features.csv` should contain features for test images.
    main()