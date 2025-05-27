import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import random
from utils.logger import setup_logger
from utils.config_parser import ConfigParser
import albumentations as A # Using Albumentations for more robust augmentations

logger = setup_logger("image_augmentor")
config = ConfigParser()

IMAGES_DIR_CONFIG_KEY = "preprocessing.augmentations.source_dir" # Key in config.yaml for source images
DEFAULT_IMAGES_DIR = os.path.join(config.get("processed_data_dir", "data/processed"), "images", "all_frames")
IMAGES_DIR = config.get(IMAGES_DIR_CONFIG_KEY, DEFAULT_IMAGES_DIR)

AUGMENTED_IMAGES_DIR_CONFIG_KEY = "preprocessing.augmentations.output_dir"
DEFAULT_AUGMENTED_IMAGES_DIR = os.path.join(config.get("processed_data_dir", "data/processed"), "images", "augmented_albumentations")
AUGMENTED_IMAGES_DIR = config.get(AUGMENTED_IMAGES_DIR_CONFIG_KEY, DEFAULT_AUGMENTED_IMAGES_DIR)

NUM_AUGMENTATIONS_PER_IMAGE = config.get("preprocessing.augmentations.num_per_image", 2)
TARGET_IMG_SIZE_CONFIG = config.get("preprocessing.augmentations.target_size", None) # e.g., [width, height] or null

# Define Albumentations transform pipeline
# Note: For object detection, geometric transforms require bbox adjustments.
# This pipeline focuses on photometric and simple geometric that might be okay without complex bbox changes,
# or assumes you'll handle bbox adjustments separately if these are used prior to annotation or with tools that support it.
# If labels are already in YOLO format (normalized), geometric transforms are more complex to update.
transform_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MotionBlur(blur_limit=7, p=0.3), # Simulates drone movement or water surface blur
    A.ImageCompression(quality_lower=85, quality_upper=95, p=0.2), # Simulates compression artifacts
    # A.RandomResizedCrop(height=TARGET_IMG_SIZE_CONFIG[1] if TARGET_IMG_SIZE_CONFIG else 720,
    #                     width=TARGET_IMG_SIZE_CONFIG[0] if TARGET_IMG_SIZE_CONFIG else 1280,
    #                     scale=(0.8, 1.0), p=0.3), # This would require bbox adjustments
    # A.Rotate(limit=15, p=0.3), # This would require bbox adjustments
])


def augment_and_save_image(image_path, output_base_dir, num_versions_to_create):
    """
    Loads an image, applies augmentations, and saves the augmented versions.
    """
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            logger.warning(f"Could not read image: {image_path}. Skipping.")
            return 0
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) # Albumentations expects RGB

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Determine subdirectory structure based on original path relative to IMAGES_DIR
        relative_dir_path = os.path.relpath(os.path.dirname(image_path), IMAGES_DIR)
        current_augmented_output_dir = os.path.join(output_base_dir, relative_dir_path)
        os.makedirs(current_augmented_output_dir, exist_ok=True)
        
        saved_count = 0
        for i in range(num_versions_to_create):
            augmented = transform_pipeline(image=original_image)
            augmented_img_rgb = augmented['image']
            augmented_img_bgr = cv2.cvtColor(augmented_img_rgb, cv2.COLOR_RGB2BGR)

            # Resize if target size is specified
            if TARGET_IMG_SIZE_CONFIG and len(TARGET_IMG_SIZE_CONFIG) == 2:
                target_w, target_h = TARGET_IMG_SIZE_CONFIG
                augmented_img_bgr = cv2.resize(augmented_img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


            aug_filename = os.path.join(current_augmented_output_dir, f"{base_filename}_aug_{i:02d}.jpg")
            cv2.imwrite(aug_filename, augmented_img_bgr)
            saved_count += 1
        return saved_count
            
    except Exception as e:
        logger.error(f"Error augmenting image {image_path}: {e}")
        return 0

def main():
    logger.info(f"Starting image augmentation from: {IMAGES_DIR}")
    logger.info(f"Output directory for augmented images: {AUGMENTED_IMAGES_DIR}")
    logger.info(f"Number of augmentations per image: {NUM_AUGMENTATIONS_PER_IMAGE}")
    if TARGET_IMG_SIZE_CONFIG:
        logger.info(f"Augmented images will be resized to: {TARGET_IMG_SIZE_CONFIG[0]}x{TARGET_IMG_SIZE_CONFIG[1]}")


    os.makedirs(AUGMENTED_IMAGES_DIR, exist_ok=True)

    # Glob images recursively from all subdirectories within IMAGES_DIR
    image_files = glob.glob(os.path.join(IMAGES_DIR, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(IMAGES_DIR, "**", "*.png"), recursive=True) + \
                  glob.glob(os.path.join(IMAGES_DIR, "**", "*.jpeg"), recursive=True)


    if not image_files:
        logger.warning(f"No image files found in {IMAGES_DIR} or its subdirectories. Exiting.")
        return

    total_augmented_created = 0
    for image_path in tqdm(image_files, desc="Augmenting Images"):
        count = augment_and_save_image(image_path, AUGMENTED_IMAGES_DIR, NUM_AUGMENTATIONS_PER_IMAGE)
        total_augmented_created += count

    logger.info(f"Image augmentation complete. Total augmented images created: {total_augmented_created}")
    logger.info("IMPORTANT: If these augmented images are used for object detection training,")
    logger.info("and geometric transformations (like crop, rotate, significant resize) were applied,")
    logger.info("the corresponding bounding box annotations MUST be updated. This script does not do that.")
    logger.info("It is generally recommended to perform augmentations that affect geometry as part of the training dataloader (e.g., in YOLOv5's dataloader).")

if __name__ == "__main__":
    main()