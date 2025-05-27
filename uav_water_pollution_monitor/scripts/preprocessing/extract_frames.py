import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import random
from utils.logger import setup_logger
from utils.config_parser import ConfigParser

# This is a basic augmentation script. For more advanced augmentations,
# consider libraries like Albumentations or imgaug.
# The paper mentions "The image augmentation pipeline ensures that the model is exposed to diverse
# and realistic conditions, improving its robustness against environmental variations."
# However, it doesn't specify which augmentations were used.
# Common augmentations for object detection:
# - Horizontal Flip
# - Brightness/Contrast/Saturation adjustments
# - Random Crop (careful with object detection, ensure RoIs are preserved or labels adjusted)
# - Noise
# - Blur

logger = setup_logger("image_augmentor")
config = ConfigParser()

IMAGES_DIR = os.path.join(config.get("processed_data_dir"), "images", "all_frames") # Or your training image dir
AUGMENTED_IMAGES_DIR = os.path.join(config.get("processed_data_dir"), "images", "augmented")
NUM_AUGMENTATIONS_PER_IMAGE = config.get("preprocessing.augmentations.num_per_image", 2) # How many augmented versions to create per original image

def augment_image(image):
    """
    Applies a set of random augmentations to an image.
    Returns a list of augmented images.
    """
    augmented_images = []

    # 1. Horizontal Flip (50% chance)
    if random.random() < 0.5:
        flipped_img = cv2.flip(image, 1)
        augmented_images.append(("flip", flipped_img))

    # 2. Brightness Adjustment (random factor)
    brightness_factor = random.uniform(0.7, 1.3) # Range 0.7 to 1.3
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_adjusted = np.clip(v * brightness_factor, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v_adjusted))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(("bright", bright_img))

    # 3. Contrast Adjustment (random factor)
    # alpha for contrast, beta for brightness (here beta=0 for only contrast)
    contrast_factor = random.uniform(0.7, 1.3)
    contrast_img = np.clip(image.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)
    augmented_images.append(("contrast", contrast_img))

    # 4. Gaussian Blur (small kernel, small sigma)
    if random.random() < 0.3: # Apply blur with 30% chance
        kernel_size = random.choice([(3,3), (5,5)])
        blur_img = cv2.GaussianBlur(image, kernel_size, 0)
        augmented_images.append(("blur", blur_img))
    
    # 5. Adding Gaussian Noise
    if random.random() < 0.3: # Apply noise with 30% chance
        row, col, ch = image.shape
        mean = 0
        sigma = random.uniform(5, 20) # Noise level
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy_img = np.clip(image + gauss, 0, 255).astype(np.uint8)
        augmented_images.append(("noise", noisy_img))

    # Select up to NUM_AUGMENTATIONS_PER_IMAGE unique augmentations
    if len(augmented_images) > NUM_AUGMENTATIONS_PER_IMAGE:
        return random.sample(augmented_images, NUM_AUGMENTATIONS_PER_IMAGE)
    return augmented_images


def main():
    logger.info(f"Starting image augmentation from: {IMAGES_DIR}")
    logger.info(f"Output directory for augmented images: {AUGMENTED_IMAGES_DIR}")
    logger.info(f"Number of augmentations per image: {NUM_AUGMENTATIONS_PER_IMAGE}")

    os.makedirs(AUGMENTED_IMAGES_DIR, exist_ok=True)

    image_files = glob.glob(os.path.join(IMAGES_DIR, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(IMAGES_DIR, "**", "*.png"), recursive=True)

    if not image_files:
        logger.warning(f"No image files found in {IMAGES_DIR} or its subdirectories. Exiting.")
        return

    total_augmented_count = 0
    for image_path in tqdm(image_files, desc="Augmenting Images"):
        try:
            original_image = cv2.imread(image_path)
            if original_image is None:
                logger.warning(f"Could not read image: {image_path}. Skipping.")
                continue

            augmented_versions = augment_image(original_image)

            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            
            # Determine subdirectory structure based on original path relative to IMAGES_DIR
            relative_path = os.path.relpath(os.path.dirname(image_path), IMAGES_DIR)
            current_augmented_dir = os.path.join(AUGMENTED_IMAGES_DIR, relative_path)
            os.makedirs(current_augmented_dir, exist_ok=True)


            for aug_type, aug_img in augmented_versions:
                aug_filename = os.path.join(current_augmented_dir, f"{base_filename}_aug_{aug_type}_{total_augmented_count}.jpg")
                cv2.imwrite(aug_filename, aug_img)
                total_augmented_count += 1

        except Exception as e:
            logger.error(f"Error augmenting image {image_path}: {e}")

    logger.info(f"Image augmentation complete. Total augmented images created: {total_augmented_count}")

if __name__ == "__main__":
    # Note: This script augments images. For object detection, if you use geometric augmentations
    # like crop, rotate, scale, you MUST ALSO TRANSFORM THE BOUNDING BOX ANNOTATIONS.
    # The current script uses photometric augmentations and horizontal flip, which might require
    # flipping x-coordinates of bounding boxes if applied to annotated data.
    # For simplicity, this script just saves augmented images. Label adjustment is complex
    # and typically handled by libraries like Albumentations when training.
    main()