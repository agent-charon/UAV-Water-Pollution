import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import torch

from utils.config_parser import ConfigParser
from utils.logger import setup_logger
from models.vit_extractor import ViTExtractor
from models.sensor_feature_loader import SensorFeatureLoader
# For YOLOv5 features, we need a way to get bounding box info.
# This implies running inference with the trained lightweight YOLOv5 model first.
# Let's assume we have a separate script/process that generates these bbox features alongside images.
# For this training script, we'll load them if available, or use placeholders.

from models.tabnet_xgb import TabNetXGBEnsemble
from utils.metrics import get_classification_metrics
from utils.visualize_results import plot_confusion_matrix

logger = setup_logger("ensemble_trainer")
config = ConfigParser()

# Paths and Parameters
PROCESSED_DATA_DIR = config.get("processed_data_dir")
IMAGES_BASE_DIR = os.path.join(PROCESSED_DATA_DIR, "yolo_dataset", "images") # Train/Val/Test splits here
LABELS_BASE_DIR = os.path.join(PROCESSED_DATA_DIR, "yolo_dataset", "labels")

# File that maps image filenames to their ground truth class labels for the ensemble model
# This needs to be created by you. Example format: image_filename,class_label_str
# e.g., train/video1_frame001.jpg,algae
IMAGE_TO_CLASS_LABEL_FILE = os.path.join(PROCESSED_DATA_DIR, "image_class_labels.csv")

# Path to store/load pre-extracted combined features
COMBINED_FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, "combined_features")
os.makedirs(COMBINED_FEATURES_DIR, exist_ok=True)

ENSEMBLE_MODEL_SAVE_PATH = os.path.join(config.get("models_dir"), "tabnet_xgb_ensemble_v1")

# --- Feature Configuration ---
# YOLOv5 bounding box features:
# For simplicity, let's assume each image has features like:
# [num_detections, avg_confidence, area_sum_norm_algae, area_sum_norm_trash, ...]
# This part is complex as it requires running YOLOv5 inference and defining how to aggregate bbox info.
# For a placeholder, let's use a fixed number of YOLO features, e.g., 4 (num_detections, total_area_norm, avg_conf, dominant_class_ confianza)
# In reality, this would come from `detect_yolov5.py` output.
# For this script, we'll assume a CSV `yolo_bbox_features.csv` exists with `image_filename` and these features.
YOLO_BBOX_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, "yolo_bbox_features.csv")
NUM_YOLO_BBOX_FEATURES = config.get("ensemble.num_yolo_bbox_features", 4) # Placeholder

def load_image_paths_and_labels(split='train'):
    """
    Loads image paths and their corresponding class labels for a given split.
    This function assumes IMAGE_TO_CLASS_LABEL_FILE exists and maps full relative image paths
    (e.g., train/img1.jpg) to class strings.
    """
    if not os.path.exists(IMAGE_TO_CLASS_LABEL_FILE):
        logger.error(f"Image to class label mapping file not found: {IMAGE_TO_CLASS_LABEL_FILE}")
        logger.error("Create this CSV with 'image_path_relative' and 'class_label' columns.")
        return [], []

    df_labels = pd.read_csv(IMAGE_TO_CLASS_LABEL_FILE)
    # Filter for the current split, assuming image_path_relative starts with 'train/', 'val/', or 'test/'
    df_split = df_labels[df_labels['image_path_relative'].str.startswith(f"{split}/")]
    
    image_paths = [os.path.join(IMAGES_BASE_DIR, rel_path) for rel_path in df_split['image_path_relative']]
    # Filter out paths that don't exist
    valid_image_paths = [p for p in image_paths if os.path.exists(p)]
    # Get corresponding labels for valid paths
    valid_labels_str = []
    for p_full in valid_image_paths:
        rel_p = os.path.relpath(p_full, IMAGES_BASE_DIR)
        label_entry = df_split[df_split['image_path_relative'] == rel_p]['class_label']
        if not label_entry.empty:
            valid_labels_str.append(label_entry.iloc[0])
        else: # Should not happen if logic is correct
            valid_image_paths.remove(p_full) # Mismatch, remove path
            logger.warning(f"Label not found for existing image {p_full}")


    if len(image_paths) != len(valid_image_paths):
        logger.warning(f"Found {len(df_split)} entries for split '{split}', but only {len(valid_image_paths)} images exist on disk.")

    return valid_image_paths, valid_labels_str


def extract_yolo_bbox_features(image_filename_base, yolo_features_df):
    """Placeholder for extracting YOLOv5 bbox features."""
    if yolo_features_df is None or yolo_features_df.empty:
        return np.full(NUM_YOLO_BBOX_FEATURES, np.nan) # Placeholder values
    
    row = yolo_features_df[yolo_features_df['image_filename'] == image_filename_base]
    if not row.empty:
        # Assuming columns like 'num_detections', 'avg_confidence', 'total_area_trash', 'total_area_algae'
        # This needs to be defined by your YOLO feature extraction process
        # For now, just take the first NUM_YOLO_BBOX_FEATURES after 'image_filename'
        try:
            return row.iloc[0, 1:NUM_YOLO_BBOX_FEATURES+1].values.astype(np.float32)
        except Exception as e:
            logger.warning(f"Could not extract specific yolo features for {image_filename_base}: {e}. Returning NaNs.")
            return np.full(NUM_YOLO_BBOX_FEATURES, np.nan)
    else:
        return np.full(NUM_YOLO_BBOX_FEATURES, np.nan)


def prepare_features_for_split(image_paths, vit_extractor, sensor_loader, yolo_features_df, split_name="train"):
    """
    Prepares combined features (ViT, YOLO, Sensor) for a list of image paths.
    """
    all_vit_features = []
    all_yolo_features = []
    all_sensor_features = []

    logger.info(f"Preparing combined features for {split_name} split ({len(image_paths)} images)...")
    for img_path in tqdm(image_paths, desc=f"Extracting features for {split_name}"):
        img_basename = os.path.basename(img_path)

        # ViT features
        vit_feat = vit_extractor.extract_features(img_path)
        if vit_feat is None: # Handle cases where ViT extraction might fail
            vit_feat = np.full(vit_extractor.get_feature_dimension(), np.nan)
        all_vit_features.append(vit_feat)

        # YOLO BBox features (placeholder)
        yolo_feat = extract_yolo_bbox_features(img_basename, yolo_features_df)
        all_yolo_features.append(yolo_feat)

        # Sensor features
        sensor_feat = sensor_loader.get_features(img_basename)
        all_sensor_features.append(sensor_feat)
        
    # Combine features
    # Ensure all features are 2D arrays before concatenating if they aren't already
    X_vit = np.array(all_vit_features)
    X_yolo = np.array(all_yolo_features)
    X_sensor = np.array(all_sensor_features)

    # Handle potential all-NaN slices from failed extractions, replace with zeros or mean
    X_vit = np.nan_to_num(X_vit, nan=0.0) # Replace NaNs with 0 for ViT
    X_yolo = np.nan_to_num(X_yolo, nan=0.0) # Replace NaNs with 0 for YOLO placeholder
    
    # For sensor, more careful NaN handling might be needed (e.g., imputation based on training data)
    # For now, also replace with 0, but StandardScaler later will help.
    X_sensor = np.nan_to_num(X_sensor, nan=0.0) 


    logger.info(f"Shapes before concat for {split_name}: ViT={X_vit.shape}, YOLO={X_yolo.shape}, Sensor={X_sensor.shape}")
    
    # Concatenate all features
    # Ensure they are all 2D before concatenating: (num_samples, num_features_per_modality)
    if X_vit.ndim == 1: X_vit = X_vit.reshape(1, -1) if X_vit.size > 0 else np.empty((1,0))
    if X_yolo.ndim == 1: X_yolo = X_yolo.reshape(1, -1) if X_yolo.size > 0 else np.empty((1,0))
    if X_sensor.ndim == 1: X_sensor = X_sensor.reshape(1, -1) if X_sensor.size > 0 else np.empty((1,0))
    
    # Handle case where one of the feature sets might be empty if no images processed
    feature_list = []
    if X_vit.shape[1] > 0: feature_list.append(X_vit)
    if X_yolo.shape[1] > 0: feature_list.append(X_yolo)
    if X_sensor.shape[1] > 0: feature_list.append(X_sensor)
    
    if not feature_list: # No features extracted at all
        return np.empty((0,0))

    X_combined = np.concatenate(feature_list, axis=1)
    logger.info(f"Combined features shape for {split_name}: {X_combined.shape}")
    
    return X_combined


def main():
    logger.info("Starting Ensemble Model Training (TabNet + XGBoost)")

    # --- 0. Initialize Feature Extractors ---
    vit_ext = ViTExtractor()
    sensor_load = SensorFeatureLoader()
    
    yolo_bbox_df = None
    if os.path.exists(YOLO_BBOX_FEATURES_FILE):
        yolo_bbox_df = pd.read_csv(YOLO_BBOX_FEATURES_FILE)
        logger.info(f"Loaded YOLO BBox features from {YOLO_BBOX_FEATURES_FILE}")
    else:
        logger.warning(f"YOLO BBox features file not found: {YOLO_BBOX_FEATURES_FILE}. YOLO features will be NaNs/zeros.")

    # --- 1. Load Data (Image Paths and Labels) ---
    # You need to create 'image_class_labels.csv'
    # Columns: image_path_relative, class_label (string)
    # e.g.: train/algae_img_001.jpg,algae
    #       train/trash_img_005.jpg,trash
    #       val/algae_img_100.jpg,algae
    # This file maps each image that will be used in the ensemble to its true class.
    if not os.path.exists(IMAGE_TO_CLASS_LABEL_FILE):
        logger.error(f"CRITICAL: {IMAGE_TO_CLASS_LABEL_FILE} not found. This file is required to map images to their classes for ensemble training.")
        logger.error("Please create it with columns: 'image_path_relative', 'class_label'")
        logger.error("Example: 'train/frame_001.jpg','algae'")
        return

    train_image_paths, train_labels_str = load_image_paths_and_labels(split='train')
    val_image_paths, val_labels_str = load_image_paths_and_labels(split='val')
    # test_image_paths, test_labels_str = load_image_paths_and_labels(split='test') # For final eval

    if not train_image_paths or not val_image_paths:
        logger.error("No training or validation image paths loaded. Check dataset and label file.")
        return

    # --- 2. Encode Labels ---
    all_labels_str = np.concatenate([train_labels_str, val_labels_str])
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels_str) # Fit on all unique string labels
    
    y_train = label_encoder.transform(train_labels_str)
    y_val = label_encoder.transform(val_labels_str)
    # y_test = label_encoder.transform(test_labels_str)
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_.tolist()
    logger.info(f"Encoded labels. Number of classes: {num_classes}, Class names: {class_names}")
    # Save label encoder
    joblib.dump(label_encoder, os.path.join(ENSEMBLE_MODEL_SAVE_PATH, "label_encoder.pkl"))


    # --- 3. Prepare/Load Combined Features ---
    # Option to load pre-extracted features to save time
    X_train_combined_path = os.path.join(COMBINED_FEATURES_DIR, "X_train_combined.npy")
    X_val_combined_path = os.path.join(COMBINED_FEATURES_DIR, "X_val_combined.npy")
    # X_test_combined_path = os.path.join(COMBINED_FEATURES_DIR, "X_test_combined.npy")

    force_reextract = config.get("ensemble.force_feature_reextraction", False)

    if not force_reextract and os.path.exists(X_train_combined_path) and os.path.exists(X_val_combined_path):
        logger.info("Loading pre-extracted combined features...")
        X_train_combined = np.load(X_train_combined_path)
        y_train_loaded = np.load(os.path.join(COMBINED_FEATURES_DIR, "y_train.npy")) # Ensure consistency
        if not np.array_equal(y_train, y_train_loaded):
            logger.warning("Loaded y_train does not match current y_train. Re-extracting features.")
            force_reextract = True
        
        X_val_combined = np.load(X_val_combined_path)
        y_val_loaded = np.load(os.path.join(COMBINED_FEATURES_DIR, "y_val.npy"))
        if not np.array_equal(y_val, y_val_loaded):
            logger.warning("Loaded y_val does not match current y_val. Re-extracting features.")
            force_reextract = True
            
    if force_reextract or not (os.path.exists(X_train_combined_path) and os.path.exists(X_val_combined_path)):
        logger.info("Extracting combined features...")
        X_train_combined = prepare_features_for_split(train_image_paths, vit_ext, sensor_load, yolo_bbox_df, "train")
        X_val_combined = prepare_features_for_split(val_image_paths, vit_ext, sensor_load, yolo_bbox_df, "val")
        # X_test_combined = prepare_features_for_split(test_image_paths, ...)
        
        np.save(X_train_combined_path, X_train_combined)
        np.save(os.path.join(COMBINED_FEATURES_DIR, "y_train.npy"), y_train)
        np.save(X_val_combined_path, X_val_combined)
        np.save(os.path.join(COMBINED_FEATURES_DIR, "y_val.npy"), y_val)
        logger.info("Saved extracted combined features.")

    if X_train_combined.shape[0] == 0 or X_val_combined.shape[0] == 0:
        logger.error("No combined features available for training or validation. Aborting.")
        return
    if X_train_combined.shape[0] != len(y_train) or X_val_combined.shape[0] != len(y_val):
        logger.error("Mismatch between number of feature samples and labels. Aborting.")
        logger.error(f"X_train: {X_train_combined.shape[0]}, y_train: {len(y_train)}")
        logger.error(f"X_val: {X_val_combined.shape[0]}, y_val: {len(y_val)}")

        return

    logger.info(f"Final shapes for training: X_train={X_train_combined.shape}, y_train={y_train.shape}")
    logger.info(f"Final shapes for validation: X_val={X_val_combined.shape}, y_val={y_val.shape}")


    # --- 4. Scale Features (Optional but recommended for TabNet/XGBoost) ---
    # TabNet might handle scaling internally, but good practice for inputs.
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train_combined)
    # X_val_scaled = scaler.transform(X_val_combined)
    # joblib.dump(scaler, os.path.join(ENSEMBLE_MODEL_SAVE_PATH, "feature_scaler.pkl"))
    # For now, let's skip explicit scaling and let TabNet/XGBoost handle it or assume features are somewhat normalized.
    # The paper doesn't explicitly mention scaling of the combined feature vector T.
    X_train_final = X_train_combined.astype(np.float32) # TabNet expects float32
    X_val_final = X_val_combined.astype(np.float32)


    # --- 5. Initialize and Train Ensemble Model ---
    input_dim = X_train_final.shape[1]
    ensemble_model = TabNetXGBEnsemble(input_dim=input_dim, output_dim=num_classes)

    # Train TabNet part
    tabnet_epochs = config.get("ensemble.tabnet_epochs", 50)
    tabnet_batch_size = config.get("ensemble.tabnet_batch_size", 1024) # TabNet often prefers larger batches
    tabnet_val_batch_size = config.get("ensemble.tabnet_virtual_batch_size", 256)

    logger.info(f"Fitting TabNet with {tabnet_epochs} epochs, batch_size={tabnet_batch_size}")
    ensemble_model.fit_tabnet(
        X_train_final, y_train,
        X_val_final, y_val,
        tabnet_fit_params={
            "max_epochs": tabnet_epochs,
            "batch_size": tabnet_batch_size,
            "virtual_batch_size": tabnet_val_batch_size,
            "eval_metric": ['mlogloss' if num_classes > 1 else 'auc'] # mlogloss for multi-class
            # Add other specific TabNet fit params from config if needed
        }
    )

    # Extract TabNet features for XGBoost training
    logger.info("Extracting TabNet features for XGBoost training...")
    X_train_tabnet_features = ensemble_model.get_tabnet_features(X_train_final)
    X_val_tabnet_features = ensemble_model.get_tabnet_features(X_val_final)

    # Train XGBoost part
    xgb_estimators = config.get("ensemble.xgb_estimators", 100)
    logger.info(f"Fitting XGBoost with {xgb_estimators} estimators...")
    ensemble_model.fit_xgboost(
        X_train_tabnet_features, y_train,
        X_val_tabnet_features, y_val,
        xgb_fit_params={
            "n_estimators": xgb_estimators,
            # Add other specific XGBoost fit params from config if needed
        }
    )

    # --- 6. Evaluate on Validation Set ---
    logger.info("Evaluating ensemble model on validation set...")
    y_val_pred_proba = ensemble_model.predict_proba(X_val_final)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)

    val_metrics = get_classification_metrics(y_val, y_val_pred, class_names=class_names, average='weighted')
    logger.info(f"Validation Metrics (Weighted):")
    logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {val_metrics['precision']:.4f}")
    logger.info(f"  Recall: {val_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {val_metrics['f1_score']:.4f}")

    val_metrics_per_class = get_classification_metrics(y_val, y_val_pred, class_names=class_names, average=None)
    logger.info(f"Validation Metrics (Per-Class):")
    for i, name in enumerate(class_names):
        logger.info(f"  Class: {name}")
        logger.info(f"    Precision: {val_metrics_per_class[f'precision_{name}']:.4f}")
        logger.info(f"    Recall: {val_metrics_per_class[f'recall_{name}']:.4f}")
        logger.info(f"    F1-Score: {val_metrics_per_class[f'f1_score_{name}']:.4f}")


    # Plot confusion matrix for validation
    cm_val_path = os.path.join(config.get("results_dir", "results"), "ensemble_val_confusion_matrix.png")
    plot_confusion_matrix(val_metrics['confusion_matrix'], class_names, title="Ensemble Validation Confusion Matrix", output_path=cm_val_path)

    # --- 7. Save the Trained Ensemble Model ---
    os.makedirs(ENSEMBLE_MODEL_SAVE_PATH, exist_ok=True)
    ensemble_model.save_model(ENSEMBLE_MODEL_SAVE_PATH)
    logger.info(f"Trained ensemble model saved to: {ENSEMBLE_MODEL_SAVE_PATH}")

    # TODO: Implement final evaluation on the test set using the saved model.

if __name__ == "__main__":
    # Before running:
    # 1. Ensure `IMAGE_TO_CLASS_LABEL_FILE` (e.g., data/processed/image_class_labels.csv) exists and is correct.
    #    It should map image filenames (relative to IMAGES_BASE_DIR) to their string class labels.
    #    The images themselves should be in IMAGES_BASE_DIR/train, IMAGES_BASE_DIR/val etc.
    # 2. Ensure `YOLO_BBOX_FEATURES_FILE` (e.g., data/processed/yolo_bbox_features.csv) exists if you want
    #    to include YOLO bbox features. It should have 'image_filename' (basename) and then feature columns.
    #    If not present, YOLO features will be zeros/NaNs.
    # 3. Ensure `SENSOR_DATA_FILE` (config.get("sensor_data_file")) exists if using sensor features.
    #    It should have 'image_filename' (basename) and sensor feature columns.
    import joblib # For saving label encoder

    # Example: Create dummy image_class_labels.csv if it doesn't exist for a quick test
    dummy_label_file = IMAGE_TO_CLASS_LABEL_FILE
    if not os.path.exists(dummy_label_file):
        logger.warning(f"Dummy {dummy_label_file} not found, creating one for a test run.")
        os.makedirs(os.path.dirname(dummy_label_file), exist_ok=True)
        dummy_img_dir_train = os.path.join(IMAGES_BASE_DIR, "train")
        dummy_img_dir_val = os.path.join(IMAGES_BASE_DIR, "val")
        os.makedirs(dummy_img_dir_train, exist_ok=True)
        os.makedirs(dummy_img_dir_val, exist_ok=True)
        
        dummy_labels_data = []
        for i in range(20): # 20 train samples
            fname = f"dummy_train_img_{i:03d}.jpg"
            # Create dummy image file
            if not os.path.exists(os.path.join(dummy_img_dir_train, fname)):
                 Image.new('RGB', (100,100)).save(os.path.join(dummy_img_dir_train, fname))
            dummy_labels_data.append([f"train/{fname}", "algae" if i % 2 == 0 else "trash"])
        for i in range(10): # 10 val samples
            fname = f"dummy_val_img_{i:03d}.jpg"
            if not os.path.exists(os.path.join(dummy_img_dir_val, fname)):
                Image.new('RGB', (100,100)).save(os.path.join(dummy_img_dir_val, fname))
            dummy_labels_data.append([f"val/{fname}", "algae" if i % 2 == 0 else "trash"])
        
        pd.DataFrame(dummy_labels_data, columns=['image_path_relative', 'class_label']).to_csv(dummy_label_file, index=False)
        logger.info(f"Created dummy {dummy_label_file} and dummy image files for testing.")
        # Also create dummy yolo_bbox_features.csv and sensor_features.csv if you want to test those paths
        # For sensor_features.csv, the sensor_feature_loader.py already creates a dummy if needed for its own test.

    from PIL import Image # For dummy image creation
    main()