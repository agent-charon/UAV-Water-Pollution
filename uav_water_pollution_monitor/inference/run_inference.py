import cv2
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from utils.config_parser import ConfigParser
from utils.logger import setup_logger
from utils.visualize_results import draw_bounding_boxes
from models.yolov5_lightweight.detect_yolov5 import YOLOv5Detector
from inference.frame_filter import PollutantFrameFilter
from inference.gps_tagger import tag_image_with_gps # Assuming GPS data is available per frame

# For multi-modal inference (optional, if classifying filtered frames)
from models.vit_extractor import ViTExtractor
from models.sensor_feature_loader import SensorFeatureLoader
from models.tabnet_xgb import TabNetXGBEnsemble
import joblib # For loading label encoder for ensemble

logger = setup_logger("inference_runner")
config = ConfigParser()

# Configuration
VIDEO_SOURCE = config.get("inference.video_source", 0) # 0 for webcam, or path to video file
OUTPUT_DIR = config.get("inference.output_dir", "inference_results")
SAVE_RELEVANT_FRAMES = config.get("inference.save_relevant_frames", True)
RELEVANT_FRAMES_SUBDIR = "relevant_pollutant_frames"
GPS_TAG_RELEVANT_FRAMES = config.get("inference.gps_tag_relevant_frames", False) # Requires GPS data stream
PERFORM_MULTIMODAL_CLASSIFICATION = config.get("inference.perform_multimodal_classification", False) # If true, classify relevant frames
FRAME_PROCESSING_INTERVAL = config.get("frame_processing_interval", 1) # Process every Nth frame

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
if SAVE_RELEVANT_FRAMES:
    os.makedirs(os.path.join(OUTPUT_DIR, RELEVANT_FRAMES_SUBDIR), exist_ok=True)

# --- Dummy GPS Data Stream (Replace with actual UAV GPS) ---
class DummyGPSSource:
    def __init__(self, start_lat=34.0522, start_lon=-118.2437, start_alt=70.0):
        self.lat = start_lat
        self.lon = start_lon
        self.alt = start_alt
        self.lat_step = 0.00001
        self.lon_step = 0.00001
        self.alt_fluctuation = 0.1

    def get_current_gps(self):
        self.lat += self.lat_step
        self.lon += self.lon_step
        self.alt += np.random.uniform(-self.alt_fluctuation, self.alt_fluctuation)
        return self.lat, self.lon, self.alt, datetime.utcnow()

def main():
    logger.info("--- Starting UAV Water Pollutant Inference Pipeline ---")

    # Initialize YOLOv5 Detector and Frame Filter
    try:
        yolo_detector = YOLOv5Detector()
        filter_relevant_classes = config.get("inference.filter_relevant_classes", ["algae", "trash"])
        frame_filter = PollutantFrameFilter(yolo_detector, relevant_classes=filter_relevant_classes)
    except Exception as e:
        logger.error(f"Failed to initialize YOLOv5 detector or FrameFilter: {e}")
        return

    # Initialize components for Multi-modal classification (if enabled)
    ensemble_model = None
    vit_ext = None
    sensor_load = None # Sensor data per frame in real-time is complex, using image-filename mapping
    yolo_bbox_features_df_inference = None # For yolo features if needed by ensemble
    label_encoder_ensemble = None

    if PERFORM_MULTIMODAL_CLASSIFICATION:
        try:
            ensemble_model_path = os.path.join(config.get("models_dir"), "tabnet_xgb_ensemble_v1")
            if not os.path.exists(ensemble_model_path):
                raise FileNotFoundError("Ensemble model directory not found.")
            
            vit_ext = ViTExtractor()
            sensor_load = SensorFeatureLoader() # This will look for sensor_features.csv

            # For YOLO features component of ensemble input (if any)
            # This is tricky for live inference. Assume we either don't use them for live classification
            # or have a way to quickly compute them. For now, we'll pass NaNs if not available.
            NUM_YOLO_BBOX_FEATURES_ENSEMBLE = config.get("ensemble.num_yolo_bbox_features", 4)


            # Load label encoder for ensemble
            label_encoder_path = os.path.join(ensemble_model_path, "label_encoder.pkl")
            if not os.path.exists(label_encoder_path):
                raise FileNotFoundError("Ensemble label encoder not found.")
            label_encoder_ensemble = joblib.load(label_encoder_path)
            num_classes_ensemble = len(label_encoder_ensemble.classes_)
            
            # Infer input_dim (this must match training)
            vit_dim = vit_ext.get_feature_dimension()
            yolo_dim_ensemble = NUM_YOLO_BBOX_FEATURES_ENSEMBLE
            sensor_dim = sensor_load.get_num_features()
            ensemble_input_dim = vit_dim + yolo_dim_ensemble + sensor_dim

            ensemble_model = TabNetXGBEnsemble(input_dim=ensemble_input_dim, output_dim=num_classes_ensemble)
            ensemble_model.load_model(ensemble_model_path)
            logger.info("Multi-modal classification components initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize multi-modal components: {e}. Disabling multi-modal classification.")
            PERFORM_MULTIMODAL_CLASSIFICATION = False


    # Initialize Video Capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error(f"Error opening video source: {VIDEO_SOURCE}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video source opened. Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}")

    # Output video writer (optional)
    # save_output_video = config.get("inference.save_output_video", False)
    # out_video = None
    # if save_output_video:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out_video_path = os.path.join(OUTPUT_DIR, f"inference_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    #     out_video = cv2.VideoWriter(out_video_path, fourcc, fps / FRAME_PROCESSING_INTERVAL if fps > 0 else 10, (frame_width, frame_height))
    #     logger.info(f"Output video will be saved to: {out_video_path}")

    gps_source = None
    if GPS_TAG_RELEVANT_FRAMES:
        gps_source = DummyGPSSource() # Replace with actual GPS interface


    frame_num = 0
    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                logger.info("End of video stream or error reading frame.")
                break

            frame_num += 1
            if frame_num % FRAME_PROCESSING_INTERVAL != 0:
                continue

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            current_frame_display = frame_bgr.copy()

            # 1. Frame Filtering (CV-only detection)
            is_relevant, cv_detections = frame_filter.is_frame_relevant(frame_bgr)
            
            # Draw CV detections on the display frame
            if cv_detections:
                current_frame_display = draw_bounding_boxes(
                    current_frame_display,
                    boxes=[d['bbox'] for d in cv_detections],
                    labels=[d['class_id'] for d in cv_detections],
                    scores=[d['confidence'] for d in cv_detections],
                    class_names=yolo_detector.get_names()
                )

            frame_info_text = f"Frame: {frame_num} | Relevant: {is_relevant}"

            if is_relevant:
                logger.info(f"Frame {frame_num}: Relevant. Pollutants detected by CV model.")
                relevant_frame_filename = f"relevant_frame_{timestamp_str}.jpg"
                relevant_frame_path = os.path.join(OUTPUT_DIR, RELEVANT_FRAMES_SUBDIR, relevant_frame_filename)

                if SAVE_RELEVANT_FRAMES:
                    cv2.imwrite(relevant_frame_path, frame_bgr) # Save original relevant frame
                    logger.debug(f"Saved relevant frame to: {relevant_frame_path}")

                    if GPS_TAG_RELEVANT_FRAMES and gps_source:
                        lat, lon, alt, ts = gps_source.get_current_gps()
                        tag_image_with_gps(relevant_frame_path, lat, lon, alt, ts)

                # 2. Multi-modal Classification (on relevant frames if enabled)
                if PERFORM_MULTIMODAL_CLASSIFICATION and ensemble_model:
                    # Prepare features for the ensemble model
                    pil_img_for_vit = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # ViT expects RGB
                    pil_img_for_vit = Image.fromarray(pil_img_for_vit)

                    vit_f_inf = vit_ext.extract_features(pil_img_for_vit)
                    if vit_f_inf is None: vit_f_inf = np.zeros(vit_ext.get_feature_dimension())
                    
                    # For sensor features, we'd need a live mapping or use a generic profile for the current frame
                    # Using the image filename convention for sensor_loader is for offline processing.
                    # For live, you might have live sensor readings from MQTT or other source.
                    # Placeholder: use average sensor values or NaNs
                    sensor_f_inf = sensor_load.get_features(relevant_frame_filename) # This will likely be NaNs if filename not in CSV
                                                                                    # Or use a default sensor profile.
                    if np.isnan(sensor_f_inf).all(): sensor_f_inf = np.zeros(sensor_load.get_num_features())


                    # YOLO features for ensemble:
                    # Can re-use cv_detections to compute aggregate YOLO features if needed.
                    # E.g., num_cv_detections = len(cv_detections)
                    # For simplicity, using NaNs here as live yolo feature agg for ensemble is complex.
                    yolo_f_inf = np.full(NUM_YOLO_BBOX_FEATURES_ENSEMBLE, np.nan)


                    combined_f_inf = np.concatenate([
                        np.nan_to_num(vit_f_inf),
                        np.nan_to_num(yolo_f_inf),
                        np.nan_to_num(sensor_f_inf)
                    ]).astype(np.float32).reshape(1, -1)

                    if combined_f_inf.shape[1] == ensemble_input_dim:
                        ensemble_pred_proba = ensemble_model.predict_proba(combined_f_inf)
                        ensemble_pred_class_idx = np.argmax(ensemble_pred_proba, axis=1)[0]
                        ensemble_pred_class_name = label_encoder_ensemble.classes_[ensemble_pred_class_idx]
                        ensemble_pred_conf = ensemble_pred_proba[0, ensemble_pred_class_idx]
                        
                        logger.info(f"  Multi-modal Classification: {ensemble_pred_class_name} (Conf: {ensemble_pred_conf:.2f})")
                        frame_info_text += f" | Ensemble: {ensemble_pred_class_name} ({ensemble_pred_conf:.2f})"
                    else:
                        logger.warning(f"  Multi-modal: Feature dimension mismatch ({combined_f_inf.shape[1]} vs {ensemble_input_dim}). Skipping classification.")
            else: # Not relevant
                logger.debug(f"Frame {frame_num}: Not relevant.")
            
            # Display info on frame
            cv2.putText(current_frame_display, frame_info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("UAV Water Pollutant Detection", current_frame_display)

            # if save_output_video and out_video:
            #     out_video.write(current_frame_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quitting inference loop.")
                break
    
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        # if out_video:
        #     out_video.release()
        cv2.destroyAllWindows()
        logger.info("Inference pipeline finished.")


if __name__ == "__main__":
    # Ensure necessary models are trained and paths in config.yaml are correct.
    # For multi-modal, an ensemble model and label encoder must exist.
    # `image_class_labels.csv` is not directly used by this live inference script for GT,
    # but sensor_loader might use its `image_filename` mapping if sensor data is tied to specific files.
    from PIL import Image # For multi-modal inference part
    main()