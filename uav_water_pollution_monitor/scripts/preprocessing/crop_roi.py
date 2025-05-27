import cv2
import os
import json 
import glob
from tqdm import tqdm
from utils.logger import setup_logger
from utils.config_parser import ConfigParser

logger = setup_logger("roi_cropper")
config = ConfigParser()

# Path to images that have corresponding annotations
# Key in config.yaml for source images for ROI cropping
IMAGES_FOR_ROI_CROP_CONFIG_KEY = "preprocessing.roi_crop.source_dir"
DEFAULT_IMAGES_FOR_ROI_CROP_DIR = os.path.join(config.get("processed_data_dir", "data/processed"), "yolo_dataset", "images", "train") # Example: crop from train set
IMAGES_FOR_ROI_CROP_DIR = config.get(IMAGES_FOR_ROI_CROP_CONFIG_KEY, DEFAULT_IMAGES_FOR_ROI_CROP_DIR)

# Path to annotations (YOLO format: .txt files)
# Key in config.yaml for corresponding annotations
ANNOTATIONS_FOR_ROI_CROP_CONFIG_KEY = "preprocessing.roi_crop.annotations_dir"
DEFAULT_ANNOTATIONS_FOR_ROI_CROP_DIR = os.path.join(config.get("processed_data_dir", "data/processed"), "yolo_dataset", "labels", "train") # Example: annotations for train set
ANNOTATIONS_FOR_ROI_CROP_DIR = config.get(ANNOTATIONS_FOR_ROI_CROP_CONFIG_KEY, DEFAULT_ANNOTATIONS_FOR_ROI_CROP_DIR)


# Output directory for cropped ROIs
CROPPED_ROIS_OUTPUT_DIR_CONFIG_KEY = "preprocessing.roi_crop.output_dir"
DEFAULT_CROPPED_ROIS_DIR = os.path.join(config.get("processed_data_dir", "data/processed"), "rois_cropped")
CROPPED_ROIS_DIR = config.get(CROPPED_ROIS_OUTPUT_DIR_CONFIG_KEY, DEFAULT_CROPPED_ROIS_DIR)

# Optional: Pad ROIs slightly
ROI_PADDING_PIXELS = config.get("preprocessing.roi_crop.padding_pixels", 0)


def crop_rois_from_image_yolo_format(image_path, yolo_annotation_path, output_base_dir, padding=0):
    """
    Crops ROIs from an image based on its YOLO format annotation file.

    Args:
        image_path (str): Path to the image file.
        yolo_annotation_path (str): Path to the YOLO .txt annotation file.
        output_base_dir (str): Base directory to save cropped ROIs (subdirs per class will be made).
        padding (int): Pixels to pad around the ROI before cropping.

    Returns:
        int: Number of ROIs successfully cropped and saved.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Could not read image: {image_path}")
        return 0
    
    H, W, _ = image.shape
    cropped_count = 0
    image_basename_no_ext = os.path.splitext(os.path.basename(image_path))[0]

    if not os.path.exists(yolo_annotation_path):
        logger.debug(f"Annotation file not found for {image_basename_no_ext}: {yolo_annotation_path}")
        return 0

    with open(yolo_annotation_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                parts = line.strip().split()
                if len(parts) != 5:
                    logger.warning(f"Skipping malformed line in {yolo_annotation_path}: '{line.strip()}'")
                    continue
                
                class_id = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])

                # Convert normalized to absolute pixel values
                box_w_abs = width_norm * W
                box_h_abs = height_norm * H
                x_min_abs = int((x_center_norm * W) - (box_w_abs / 2))
                y_min_abs = int((y_center_norm * H) - (box_h_abs / 2))
                x_max_abs = int(x_min_abs + box_w_abs)
                y_max_abs = int(y_min_abs + box_h_abs)

                # Add padding
                x_min_padded = max(0, x_min_abs - padding)
                y_min_padded = max(0, y_min_abs - padding)
                x_max_padded = min(W, x_max_abs + padding)
                y_max_padded = min(H, y_max_abs + padding)

                if x_max_padded > x_min_padded and y_max_padded > y_min_padded:
                    roi = image[y_min_padded:y_max_padded, x_min_padded:x_max_padded]
                    
                    # Create subdirectory for this class ID if it doesn't exist
                    class_specific_output_dir = os.path.join(output_base_dir, f"class_{class_id}")
                    os.makedirs(class_specific_output_dir, exist_ok=True)
                    
                    roi_filename = os.path.join(class_specific_output_dir, f"{image_basename_no_ext}_roi_{i:03d}_cls{class_id}.jpg")
                    cv2.imwrite(roi_filename, roi)
                    cropped_count += 1
            except Exception as e:
                logger.error(f"Error processing ROI {i} in {yolo_annotation_path} for image {image_path}: {e}")
    
    if cropped_count > 0:
        logger.debug(f"Cropped {cropped_count} ROIs from {image_basename_no_ext}")
    return cropped_count


def main():
    logger.info("Starting ROI cropping from YOLO annotated images.")
    logger.info(f"Reading images from: {IMAGES_FOR_ROI_CROP_DIR}")
    logger.info(f"Reading corresponding YOLO annotations from: {ANNOTATIONS_FOR_ROI_CROP_DIR}")
    logger.info(f"Saving cropped ROIs to: {CROPPED_ROIS_DIR} (with subdirs per class)")
    logger.info(f"ROI Padding: {ROI_PADDING_PIXELS} pixels")

    os.makedirs(CROPPED_ROIS_DIR, exist_ok=True)

    # Assume image files and annotation files have the same base name and are in parallel directories
    image_files = glob.glob(os.path.join(IMAGES_FOR_ROI_CROP_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(IMAGES_FOR_ROI_CROP_DIR, "*.png")) + \
                  glob.glob(os.path.join(IMAGES_FOR_ROI_CROP_DIR, "*.jpeg"))


    if not image_files:
        logger.warning(f"No image files found in {IMAGES_FOR_ROI_CROP_DIR}. Exiting ROI cropping.")
        return

    total_rois_cropped_all_images = 0
    for image_file_path in tqdm(image_files, desc="Cropping ROIs"):
        image_filename_with_ext = os.path.basename(image_file_path)
        image_name_no_ext = os.path.splitext(image_filename_with_ext)[0]

        # Corresponding YOLO annotation file path
        annotation_file_path = os.path.join(ANNOTATIONS_FOR_ROI_CROP_DIR, f"{image_name_no_ext}.txt")

        if not os.path.exists(annotation_file_path):
            logger.debug(f"Annotation file not found for {image_filename_with_ext} at {annotation_file_path}. Skipping ROI crop for this image.")
            continue
        
        rois_count_for_image = crop_rois_from_image_yolo_format(
            image_file_path, 
            annotation_file_path, 
            CROPPED_ROIS_DIR,
            padding=ROI_PADDING_PIXELS
        )
        total_rois_cropped_all_images += rois_count_for_image

    logger.info(f"ROI cropping complete. Total ROIs cropped from all images: {total_rois_cropped_all_images}")
    logger.info(f"Cropped ROIs are saved in class-specific subdirectories under: {CROPPED_ROIS_DIR}")

if __name__ == "__main__":
    # Before running this:
    # 1. Ensure IMAGES_FOR_ROI_CROP_DIR contains the images you want to crop from.
    # 2. Ensure ANNOTATIONS_FOR_ROI_CROP_DIR contains the corresponding YOLO .txt annotation files.
    #    The script expects that if `img1.jpg` is in IMAGES_FOR_ROI_CROP_DIR, then `img1.txt` is in ANNOTATIONS_FOR_ROI_CROP_DIR.
    main()