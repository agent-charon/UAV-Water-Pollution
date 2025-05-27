import time
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch

from utils.config_parser import ConfigParser
from utils.logger import setup_logger
# For CV-only (Lightweight YOLOv5)
from models.yolov5_lightweight.detect_yolov5 import YOLOv5Detector # We'll create this in inference
# For Multi-modal
from models.vit_extractor import ViTExtractor
from models.sensor_feature_loader import SensorFeatureLoader
from models.tabnet_xgb import TabNetXGBEnsemble

# This script benchmarks inference time for both CV-only and Multi-modal approaches.
# It simulates processing a stream of frames.

logger = setup_logger("real_time_benchmarker")
config = ConfigParser()

# Parameters
NUM_BENCHMARK_FRAMES = config.get("benchmark.num_frames", 100)
DUMMY_IMAGE_DIR = os.path.join(config.get("processed_data_dir", "data/processed"), "yolo_dataset", "images", "val") # Use some val images for benchmark
YOLO_MODEL_WEIGHTS = config.get("yolov5_weights_path")
ENSEMBLE_MODEL_PATH = os.path.join(config.get("models_dir"), "tabnet_xgb_ensemble_v1")
IMG_WIDTH = config.get("image_width", 1280)
IMG_HEIGHT = config.get("image_height", 720)

def create_dummy_frame():
    return np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

def benchmark_cv_only():
    logger.info("--- Benchmarking CV-Only (Lightweight YOLOv5) Inference Time ---")
    if not YOLO_MODEL_WEIGHTS or not os.path.exists(YOLO_MODEL_WEIGHTS):
        logger.error(f"YOLOv5 weights not found at {YOLO_MODEL_WEIGHTS}. Cannot benchmark.")
        return None
    
    try:
        # Initialize YOLOv5 detector (from the inference script we will create)
        # We need detect_yolov5.py to be implemented first.
        # For now, let's assume it exists and has a simple interface.
        # from models.yolov5_lightweight.detect_yolov5 import YOLOv5Detector # Defined in inference
        # This creates a circular dependency if detect_yolov5.py also uses this benchmark.
        # Let's simulate its core logic here for benchmark.
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device} for YOLOv5 benchmark")
        # This part would typically load the model using Ultralytics Hub or custom load
        # For benchmark, we can use a simplified model loading if detect_yolov5.py isn't ready.
        # Using torch.hub.load for simplicity, assuming custom model cfg is compatible
        try:
            model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_WEIGHTS, device=device, force_reload=False) # Use cfg from model for custom
            model_yolo.conf = config.get("inference_confidence_threshold", 0.4) # from config
            model_yolo.iou = config.get("iou_threshold", 0.5) # from config

        except Exception as e:
            logger.error(f"Failed to load YOLOv5 model for benchmark via torch.hub: {e}")
            logger.info("Ensure your custom model cfg is compatible or implement direct loading.")
            return None

        logger.info(f"YOLOv5 model loaded for benchmark. Input size expected by model: {model_yolo.imgsz}")


        total_time = 0
        image_files = [os.path.join(DUMMY_IMAGE_DIR, f) for f in os.listdir(DUMMY_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            logger.warning(f"No images found in {DUMMY_IMAGE_DIR} for benchmark. Using dummy frames.")
            frames_to_process = [create_dummy_frame() for _ in range(NUM_BENCHMARK_FRAMES)]
        else:
            frames_to_process = [cv2.imread(image_files[i % len(image_files)]) for i in range(NUM_BENCHMARK_FRAMES)]


        # Warm-up run
        logger.info("Performing warm-up run for YOLOv5...")
        if isinstance(frames_to_process[0], np.ndarray):
             _ = model_yolo(frames_to_process[0][:,:,::-1]) # BGR to RGB for PIL-based models like hub

        logger.info(f"Benchmarking YOLOv5 on {len(frames_to_process)} frames...")
        for frame_bgr in tqdm(frames_to_process, desc="YOLOv5 Benchmark"):
            start_t = time.perf_counter()
            # YOLOv5 expects RGB images or paths. If frame is BGR (OpenCV), convert.
            _ = model_yolo(frame_bgr[:,:,::-1]) # Convert BGR to RGB
            end_t = time.perf_counter()
            total_time += (end_t - start_t)

        avg_time_cv = total_time / len(frames_to_process)
        fps_cv = 1.0 / avg_time_cv if avg_time_cv > 0 else 0
        logger.info(f"CV-Only Average Inference Time: {avg_time_cv*1000:.2f} ms")
        logger.info(f"CV-Only Estimated FPS: {fps_cv:.2f}")
        return avg_time_cv

    except Exception as e:
        logger.error(f"Error during CV-Only benchmark: {e}")
        return None


def benchmark_multimodal():
    logger.info("--- Benchmarking Multi-Modal Ensemble Inference Time ---")
    if not os.path.exists(ENSEMBLE_MODEL_PATH):
        logger.error(f"Ensemble model not found at {ENSEMBLE_MODEL_PATH}. Cannot benchmark.")
        return None

    try:
        # Initialize components
        vit_ext = ViTExtractor()
        sensor_load = SensorFeatureLoader() # Will use dummy data if file not found
        # For YOLO features in multi-modal, simulate extraction or use pre-saved
        yolo_features_df = None
        if os.path.exists(config.get("evaluation.yolo_bbox_features_for_benchmark_csv", "dummy_yolo_benchmark_feats.csv")):
             yolo_bbox_df = pd.read_csv(config.get("evaluation.yolo_bbox_features_for_benchmark_csv"))
        else: # Create dummy yolo features for benchmark run
            dummy_yolo_data = {'image_filename': [f'dummy_frame_{i}.jpg' for i in range(NUM_BENCHMARK_FRAMES)],
                               **{f'yolo_feat_{j}': np.random.rand(NUM_BENCHMARK_FRAMES) for j in range(config.get("ensemble.num_yolo_bbox_features",4))}}
            yolo_bbox_df = pd.DataFrame(dummy_yolo_data)


        num_classes_config = len(config.get("classes", ["algae", "trash", "plastic", "chemical_spill", "mixed_pollutants"]))
        # Infer input_dim for ensemble (this is tricky without a sample run)
        # Let's assume it's known from training or can be estimated
        vit_dim = vit_ext.get_feature_dimension()
        yolo_dim = config.get("ensemble.num_yolo_bbox_features", 4)
        sensor_dim = sensor_load.get_num_features()
        ensemble_input_dim = vit_dim + yolo_dim + sensor_dim

        ensemble_model = TabNetXGBEnsemble(input_dim=ensemble_input_dim, output_dim=num_classes_config)
        ensemble_model.load_model(ENSEMBLE_MODEL_PATH)
        logger.info("Multi-modal ensemble model loaded for benchmark.")

        total_time = 0
        
        image_files = [os.path.join(DUMMY_IMAGE_DIR, f) for f in os.listdir(DUMMY_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            logger.warning(f"No images found in {DUMMY_IMAGE_DIR} for multimodal benchmark. Using dummy frames.")
            frames_to_process_pil = [Image.fromarray(create_dummy_frame()[:,:,::-1]) for _ in range(NUM_BENCHMARK_FRAMES)] # PIL for ViT
        else:
            frames_to_process_pil = [Image.open(image_files[i % len(image_files)]).convert('RGB') for i in range(NUM_BENCHMARK_FRAMES)]


        # Warm-up run
        logger.info("Performing warm-up run for Multi-modal ensemble...")
        img_pil = frames_to_process_pil[0]
        img_basename = f"dummy_frame_0.jpg" # For sensor/yolo lookup
        
        vit_feat_w = vit_ext.extract_features(img_pil)
        if vit_feat_w is None: vit_feat_w = np.zeros(vit_dim)
        yolo_feat_w = extract_yolo_bbox_features(img_basename, yolo_bbox_df) # Using helper from train_ensemble
        sensor_feat_w = sensor_load.get_features(img_basename)
        
        combined_feat_w = np.concatenate([
            np.nan_to_num(vit_feat_w), 
            np.nan_to_num(yolo_feat_w), 
            np.nan_to_num(sensor_feat_w)
        ]).astype(np.float32).reshape(1, -1)
        if combined_feat_w.shape[1] != ensemble_input_dim:
             logger.warning(f"Warmup feature dim {combined_feat_w.shape[1]} != expected {ensemble_input_dim}. Check feature extraction.")
        else:
            _ = ensemble_model.predict_proba(combined_feat_w)


        logger.info(f"Benchmarking Multi-modal ensemble on {len(frames_to_process_pil)} frames...")
        for i, frame_pil in tqdm(enumerate(frames_to_process_pil), total=len(frames_to_process_pil), desc="Multi-modal Benchmark"):
            start_t = time.perf_counter()
            
            current_img_basename = f"dummy_frame_{i}.jpg" # For sensor/yolo data lookup
            
            # 1. ViT features
            vit_f = vit_ext.extract_features(frame_pil)
            if vit_f is None: vit_f = np.zeros(vit_dim) # Handle potential errors

            # 2. YOLO features (simulated/loaded)
            yolo_f = extract_yolo_bbox_features(current_img_basename, yolo_bbox_df)

            # 3. Sensor features
            sensor_f = sensor_load.get_features(current_img_basename)

            # Combine
            combined_f = np.concatenate([
                np.nan_to_num(vit_f), 
                np.nan_to_num(yolo_f), 
                np.nan_to_num(sensor_f)
            ]).astype(np.float32).reshape(1, -1)

            if combined_f.shape[1] != ensemble_input_dim:
                logger.debug(f"Feature dim mismatch: got {combined_f.shape[1]}, expected {ensemble_input_dim}. Using zeros for this frame.")
                combined_f = np.zeros((1, ensemble_input_dim), dtype=np.float32)


            # 4. Ensemble prediction
            _ = ensemble_model.predict_proba(combined_f)
            
            end_t = time.perf_counter()
            total_time += (end_t - start_t)

        avg_time_mm = total_time / len(frames_to_process_pil)
        fps_mm = 1.0 / avg_time_mm if avg_time_mm > 0 else 0
        logger.info(f"Multi-Modal Average Inference Time: {avg_time_mm*1000:.2f} ms")
        logger.info(f"Multi-Modal Estimated FPS: {fps_mm:.2f}")
        return avg_time_mm

    except Exception as e:
        logger.error(f"Error during Multi-Modal benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

# Re-using this helper from train_ensemble.py for consistency
def extract_yolo_bbox_features(image_filename_base, yolo_features_df):
    NUM_YOLO_BBOX_FEATURES_BENCH = config.get("ensemble.num_yolo_bbox_features", 4)
    if yolo_features_df is None or yolo_features_df.empty:
        return np.full(NUM_YOLO_BBOX_FEATURES_BENCH, np.nan)
    
    row = yolo_features_df[yolo_features_df['image_filename'] == image_filename_base]
    if not row.empty:
        try:
            return row.iloc[0, 1:NUM_YOLO_BBOX_FEATURES_BENCH+1].values.astype(np.float32)
        except Exception:
            return np.full(NUM_YOLO_BBOX_FEATURES_BENCH, np.nan)
    else:
        return np.full(NUM_YOLO_BBOX_FEATURES_BENCH, np.nan)


if __name__ == "__main__":
    from PIL import Image # For multimodal benchmark
    import pandas as pd # For multimodal benchmark yolo features

    # Ensure DUMMY_IMAGE_DIR has some images, or dummy frames will be used.
    if not os.path.exists(DUMMY_IMAGE_DIR) or not os.listdir(DUMMY_IMAGE_DIR):
        os.makedirs(DUMMY_IMAGE_DIR, exist_ok=True)
        logger.warning(f"{DUMMY_IMAGE_DIR} is empty. Creating a dummy image for benchmark.")
        try:
            from PIL import Image as PILImage
            dummy_bench_img_path = os.path.join(DUMMY_IMAGE_DIR, "benchmark_dummy_img.jpg")
            if not os.path.exists(dummy_bench_img_path):
                 PILImage.new('RGB', (IMG_WIDTH, IMG_HEIGHT)).save(dummy_bench_img_path)
        except ImportError:
            logger.error("Pillow not installed, cannot create dummy image for DUMMY_IMAGE_DIR.")


    avg_time_cv_val = benchmark_cv_only()
    avg_time_mm_val = benchmark_multimodal()

    if avg_time_cv_val is not None:
        logger.info(f"\n--- CV-Only Benchmark Summary ---")
        logger.info(f"Average Inference Time: {avg_time_cv_val*1000:.2f} ms/frame")
        logger.info(f"FPS: {(1/avg_time_cv_val if avg_time_cv_val > 0 else 0):.2f}")

    if avg_time_mm_val is not None:
        logger.info(f"\n--- Multi-Modal Benchmark Summary ---")
        logger.info(f"Average Inference Time: {avg_time_mm_val*1000:.2f} ms/frame")
        logger.info(f"FPS: {(1/avg_time_mm_val if avg_time_mm_val > 0 else 0):.2f}")

    # The paper mentions (Table 3):
    # Proposed model (Lightweight YOLOv5 for Algae): ~30ms
    # Proposed model (Lightweight YOLOv5 for Trash): ~20ms
    # The multi-modal system's overall inference time for classification isn't directly in Table 3,
    # but implies feature extraction + TabNet + XGBoost.
    # Table 6 shows "Proposed model" (multi-class pollutants, presumably multi-modal) at ~55ms.
    # This script should aim to reproduce these timings.