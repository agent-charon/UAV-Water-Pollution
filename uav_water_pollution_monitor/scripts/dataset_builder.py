import os
import shutil
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.logger import setup_logger
from utils.config_parser import ConfigParser

logger = setup_logger("dataset_builder")
config = ConfigParser()

# Source of all processed (and potentially augmented) images
ALL_IMAGES_SOURCE_DIR = os.path.join(config.get("processed_data_dir"), "images", "all_frames") # Or "augmented" if using them
# Source of annotations (YOLO format: .txt files)
ALL_ANNOTATIONS_SOURCE_DIR = config.get("annotations_dir", "data/annotations/yolo_format/labels/all") # Central place for all .txt labels

# Destination for YOLOv5 formatted dataset
YOLO_DATASET_ROOT = os.path.join(config.get("processed_data_dir"), "yolo_dataset")
YOLO_IMAGES_DIR = os.path.join(YOLO_DATASET_ROOT, "images")
YOLO_LABELS_DIR = os.path.join(YOLO_DATASET_ROOT, "labels")

TRAIN_RATIO = config.get("dataset_split.train_ratio", 0.7) # Paper: 70%
VAL_RATIO = config.get("dataset_split.val_ratio", 0.1)   # Paper: 10%
TEST_RATIO = config.get("dataset_split.test_ratio", 0.2)  # Paper: 20% (derived)

RANDOM_STATE = 42

def create_yolo_dirs():
    os.makedirs(os.path.join(YOLO_IMAGES_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_IMAGES_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_IMAGES_DIR, "test"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_LABELS_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_LABELS_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_LABELS_DIR, "test"), exist_ok=True)
    logger.info(f"Created directory structure at {YOLO_DATASET_ROOT}")

def copy_files(file_list, image_source_base, label_source_base, dest_image_dir, dest_label_dir):
    """Copies image and its corresponding label file."""
    copied_count = 0
    for file_basename_no_ext in tqdm(file_list, desc=f"Copying to {os.path.basename(dest_image_dir)}"):
        # Find the image file (could be jpg, png etc.)
        img_src_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            potential_img_path = os.path.join(image_source_base, file_basename_no_ext + ext)
            if os.path.exists(potential_img_path):
                img_src_path = potential_img_path
                break
        
        label_src_path = os.path.join(label_source_base, file_basename_no_ext + ".txt")

        if img_src_path and os.path.exists(img_src_path) and os.path.exists(label_src_path):
            img_dest_path = os.path.join(dest_image_dir, os.path.basename(img_src_path))
            label_dest_path = os.path.join(dest_label_dir, os.path.basename(label_src_path))
            try:
                shutil.copy2(img_src_path, img_dest_path)
                shutil.copy2(label_src_path, label_dest_path)
                copied_count +=1
            except Exception as e:
                logger.error(f"Error copying {file_basename_no_ext}: {e}")
        else:
            if not (img_src_path and os.path.exists(img_src_path)):
                logger.warning(f"Image file not found for {file_basename_no_ext} in {image_source_base} with common extensions. Skipping.")
            if not os.path.exists(label_src_path):
                 logger.warning(f"Label file not found for {file_basename_no_ext} at {label_src_path}. Skipping.")
    return copied_count


def main():
    logger.info("Starting dataset builder for YOLOv5 format.")
    
    if not os.path.isdir(ALL_IMAGES_SOURCE_DIR):
        logger.error(f"Image source directory not found: {ALL_IMAGES_SOURCE_DIR}")
        return
    if not os.path.isdir(ALL_ANNOTATIONS_SOURCE_DIR):
        logger.error(f"Annotation source directory not found: {ALL_ANNOTATIONS_SOURCE_DIR}")
        return

    # Get a list of base filenames (without extension) from annotation files
    # This assumes that every annotation file has a corresponding image.
    annotation_files = glob.glob(os.path.join(ALL_ANNOTATIONS_SOURCE_DIR, "*.txt"))
    if not annotation_files:
        logger.error(f"No annotation .txt files found in {ALL_ANNOTATIONS_SOURCE_DIR}. Cannot build dataset.")
        return

    basenames = sorted([os.path.splitext(os.path.basename(f))[0] for f in annotation_files])
    logger.info(f"Found {len(basenames)} unique annotated samples (based on .txt files).")

    # Check ratios
    if not np.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
        logger.error("Train, validation, and test ratios must sum to 1.0.")
        return

    # Split data
    # First split: train_val vs test
    if TEST_RATIO > 0:
        train_val_files, test_files = train_test_split(basenames, test_size=TEST_RATIO, random_state=RANDOM_STATE)
    else:
        train_val_files = basenames
        test_files = []

    # Second split: train vs val (from train_val_files)
    # Adjust val_ratio relative to the size of train_val_files
    if VAL_RATIO > 0 and len(train_val_files) > 0:
        relative_val_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        train_files, val_files = train_test_split(train_val_files, test_size=relative_val_ratio, random_state=RANDOM_STATE)
    else:
        train_files = train_val_files
        val_files = []


    logger.info(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test samples.")

    # Create directories for YOLO dataset
    create_yolo_dirs()

    # Copy files to their respective directories
    logger.info("Copying training files...")
    train_copied = copy_files(train_files, ALL_IMAGES_SOURCE_DIR, ALL_ANNOTATIONS_SOURCE_DIR,
               os.path.join(YOLO_IMAGES_DIR, "train"), os.path.join(YOLO_LABELS_DIR, "train"))
    
    logger.info("Copying validation files...")
    val_copied = copy_files(val_files, ALL_IMAGES_SOURCE_DIR, ALL_ANNOTATIONS_SOURCE_DIR,
             os.path.join(YOLO_IMAGES_DIR, "val"), os.path.join(YOLO_LABELS_DIR, "val"))
    
    logger.info("Copying testing files...")
    test_copied = copy_files(test_files, ALL_IMAGES_SOURCE_DIR, ALL_ANNOTATIONS_SOURCE_DIR,
              os.path.join(YOLO_IMAGES_DIR, "test"), os.path.join(YOLO_LABELS_DIR, "test"))

    logger.info(f"Successfully copied: {train_copied} train, {val_copied} val, {test_copied} test files.")
    logger.info(f"YOLO dataset created at: {YOLO_DATASET_ROOT}")
    logger.info(f"Ensure your '{config.get('yolov5_dataset_yaml')}' points to '{YOLO_DATASET_ROOT}' and has correct class names/nc.")

if __name__ == "__main__":
    import numpy as np # Add this for np.isclose
    # Important: Before running, ensure:
    # 1. ALL_IMAGES_SOURCE_DIR contains all your images (e.g., from extract_frames.py or augmentations.py).
    #    The script expects image files directly in this directory or in subdirs if your annotation paths reflect that.
    #    Currently, it looks for images directly in ALL_IMAGES_SOURCE_DIR. Modify if your structure is nested.
    # 2. ALL_ANNOTATIONS_SOURCE_DIR contains all corresponding YOLO .txt label files.
    #    The name of the .txt file (e.g., 'image1.txt') must match the image name (e.g., 'image1.jpg').
    # This script assumes a flat structure in ALL_IMAGES_SOURCE_DIR and ALL_ANNOTATIONS_SOURCE_DIR for simplicity.
    # If your images are in subdirectories (e.g. .../all_frames/video1_frames/frame_001.jpg),
    # then ALL_ANNOTATIONS_SOURCE_DIR should also mirror that structure for labels,
    # or the `copy_files` function and `basenames` collection logic needs to be adapted.
    # The current `copy_files` assumes images are directly in `image_source_base` for a given basename.

    # A better approach for `basenames` collection if images/labels are in nested subdirs:
    # 1. Glob all .txt files from ALL_ANNOTATIONS_SOURCE_DIR recursively.
    # 2. For each .txt file, determine its relative path from ALL_ANNOTATIONS_SOURCE_DIR.
    # 3. The corresponding image should have the same relative path (with different extension) from ALL_IMAGES_SOURCE_DIR.
    # The `copy_files` function would then need to use these relative paths.
    # For now, keeping it simple: assumes all images in ALL_IMAGES_SOURCE_DIR and labels in ALL_ANNOTATIONS_SOURCE_DIR.
    main()