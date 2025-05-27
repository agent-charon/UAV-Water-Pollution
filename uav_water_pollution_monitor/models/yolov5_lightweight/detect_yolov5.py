import torch
import numpy as np
import cv2
import os
from utils.config_parser import ConfigParser
from utils.logger import setup_logger

# This script provides a class to use a trained YOLOv5 model for detection.
# It leverages torch.hub.load for loading the model.

logger = setup_logger("yolov5_detector")
config_global = ConfigParser() # Global config for default paths if not provided

class YOLOv5Detector:
    def __init__(self, weights_path=None, conf_thresh=None, iou_thresh=None, device=None, imgsz=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"YOLOv5Detector using device: {self.device}")

        self.weights_path = weights_path if weights_path else config_global.get("yolov5_weights_path")
        if not self.weights_path or not os.path.exists(self.weights_path):
            logger.error(f"YOLOv5 weights not found at: {self.weights_path}")
            # Attempt to find in default run location
            latest_run_dir = os.path.join("uav_water_pollutant_runs", config_global.get("yolov5_params.experiment_name", "lightweight_yolov5_custom_run"))
            potential_weights_path = os.path.join(latest_run_dir, "weights", "best.pt")
            if os.path.exists(potential_weights_path):
                logger.info(f"Found potential weights at {potential_weights_path}, using this.")
                self.weights_path = potential_weights_path
            else:
                 raise FileNotFoundError(f"YOLOv5 weights not found at {self.weights_path} or default run location.")

        self.conf_thresh = conf_thresh if conf_thresh is not None else config_global.get("inference_confidence_threshold", 0.4)
        self.iou_thresh = iou_thresh if iou_thresh is not None else config_global.get("iou_threshold", 0.45)
        
        # Default image size for inference. Can be overridden.
        # The model was trained on a certain size (e.g. 1280 from paper)
        self.imgsz = imgsz if imgsz is not None else config_global.get("inference.yolo_imgsz", config_global.get("image_width", 1280))

        try:
            # Load the custom model.
            # 'force_reload=True' can be useful if the model definition changes often during dev.
            # Set to False for production or stable models.
            # The 'custom' flag tells torch.hub to look for a model definition (cfg) associated with the weights,
            # or you can specify cfg directly if needed.
            # If your `model.yaml` is in the yolov5 repo structure correctly referenced by the .pt file, this should work.
            # Otherwise, you might need to load the model with a specific cfg path.
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights_path, device=self.device, force_reload=False)
            
            # For a truly custom model.yaml NOT in the standard yolov5 path:
            # self.model = torch.hub.load('path/to/your/cloned/yolov5_repo', 'custom',
            #                             path=self.weights_path, source='local',
            #                             cfg='path/to/your/models/yolov5_lightweight/model.yaml')

            self.model.conf = self.conf_thresh
            self.model.iou = self.iou_thresh
            self.model.agnostic = False # NMS class-agnostic, False by default
            self.model.multi_label = False # Multiple labels per box, False by default
            self.model.max_det = config_global.get("inference.yolo_max_det", 1000) # Max detections per image

            logger.info(f"YOLOv5 model loaded from {self.weights_path}")
            logger.info(f"Confidence threshold: {self.model.conf}, IoU threshold: {self.model.iou}, ImgSz: {self.imgsz}")
            self.model.eval()

        except Exception as e:
            logger.error(f"Error loading YOLOv5 model: {e}")
            logger.error("Ensure 'ultralytics/yolov5' is accessible (internet or local clone with `source='local'`)")
            logger.error("And that the weights path and custom model configuration are correct.")
            raise

    def detect(self, image_bgr_or_path):
        """
        Performs object detection on a single image.

        Args:
            image_bgr_or_path (np.array or str): Image in BGR format (H,W,C) or path to an image file.

        Returns:
            list: A list of detections. Each detection is a dictionary:
                  {'bbox': [x_min, y_min, x_max, y_max], 'class_id': int, 'class_name': str, 'confidence': float}
                  Returns empty list if no detections or an error occurs.
        """
        try:
            if isinstance(image_bgr_or_path, str):
                if not os.path.exists(image_bgr_or_path):
                    logger.error(f"Image path not found: {image_bgr_or_path}")
                    return []
                img_input = image_bgr_or_path # Pass path directly
            elif isinstance(image_bgr_or_path, np.ndarray):
                img_input = image_bgr_or_path[:, :, ::-1] # Convert BGR to RGB
            else:
                logger.error("Invalid input type for detection. Must be image path or BGR NumPy array.")
                return []

            results = self.model(img_input, size=self.imgsz)

            detections = []
            # results.xyxy is a list (per image in batch) of tensors [xmin, ymin, xmax, ymax, conf, class]
            # For single image, access results.xyxy[0]
            if results.xyxy[0].shape[0] > 0:
                for det in results.xyxy[0].cpu().numpy():
                    x_min, y_min, x_max, y_max, conf, cls_id = det
                    detections.append({
                        'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                        'class_id': int(cls_id),
                        'class_name': self.model.names[int(cls_id)],
                        'confidence': float(conf)
                    })
            return detections

        except Exception as e:
            logger.error(f"Error during YOLOv5 detection: {e}")
            return []

    def get_names(self):
        return self.model.names if hasattr(self.model, 'names') else []

if __name__ == '__main__':
    from utils.visualize_results import draw_bounding_boxes
    # This test requires a trained .pt file.
    # Ensure config.yaml has `yolov5_weights_path` pointing to your trained lightweight model.
    
    # Create a dummy image for testing if no other images are available
    dummy_test_img_path = "temp_yolo_detector_test.jpg"
    if not os.path.exists(dummy_test_img_path):
        try:
            from PIL import Image as PILImage, ImageDraw
            img_pil = PILImage.new('RGB', (640, 480), color = 'skyblue')
            draw = ImageDraw.Draw(img_pil)
            draw.rectangle(((100,100),(200,200)), fill="green", outline="black") # Simulate an object
            img_pil.save(dummy_test_img_path)
            logger.info(f"Created dummy image {dummy_test_img_path} for testing detector.")
        except ImportError:
            logger.error("Pillow not installed, cannot create dummy image for testing.")
            dummy_test_img_path = None
        except Exception as e:
            logger.error(f"Error creating dummy image: {e}")
            dummy_test_img_path = None


    try:
        detector = YOLOv5Detector(imgsz=640) # Override imgsz for testing if needed
        logger.info(f"Class names from model: {detector.get_names()}")

        if dummy_test_img_path and os.path.exists(dummy_test_img_path):
            image_to_test = dummy_test_img_path
            img_bgr = cv2.imread(image_to_test)
            if img_bgr is None:
                logger.error(f"Failed to read image {image_to_test} with OpenCV.")
            else:
                detections = detector.detect(img_bgr) # Test with BGR image
                # detections = detector.detect(image_to_test) # Test with image path

                logger.info(f"Detections found: {len(detections)}")
                for det in detections:
                    logger.info(f"  {det['class_name']} (ID: {det['class_id']}) at {det['bbox']} with conf: {det['confidence']:.2f}")

                # Visualize
                if detections:
                    img_vis = draw_bounding_boxes(
                        img_bgr,
                        boxes=[d['bbox'] for d in detections],
                        labels=[d['class_id'] for d in detections],
                        scores=[d['confidence'] for d in detections],
                        class_names=detector.get_names()
                    )
                    cv2.imshow("YOLOv5 Detections", img_vis)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    # cv2.imwrite("temp_yolo_detector_output.jpg", img_vis)
                    # logger.info("Detection result saved to temp_yolo_detector_output.jpg")
        else:
            logger.warning("No image available to test YOLOv5 detector.")
            # Test with a dummy numpy array
            dummy_np_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections_np = detector.detect(dummy_np_array)
            logger.info(f"Detections on random NumPy array: {len(detections_np)}")


    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError during detector init: {e}. Ensure weights path is correct.")
    except Exception as e:
        logger.error(f"An error occurred in YOLOv5Detector example: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(dummy_test_img_path):
             pass # os.remove(dummy_test_img_path)