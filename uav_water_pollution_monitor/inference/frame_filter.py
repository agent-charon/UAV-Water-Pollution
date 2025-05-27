import os
from utils.config_parser import ConfigParser
from utils.logger import setup_logger

# This module implements the logic from paper Section 4.3 / Figure 9:
# "The lightweight model, processed this video and retained only pollutant-relevant frames...
#  This not only reduced storage demands but also optimized energy usage..."
# This implies running a quick detection; if pollutants are found, the frame is "relevant".

logger = setup_logger("frame_filter")
config = ConfigParser()

class PollutantFrameFilter:
    def __init__(self, yolo_detector, relevance_threshold=None, relevant_classes=None):
        """
        Initializes the frame filter.

        Args:
            yolo_detector: An instance of YOLOv5Detector (or a similar detector).
            relevance_threshold (float, optional): Minimum confidence for a detection to be considered.
                                                  Defaults to yolo_detector's conf_thresh.
            relevant_classes (list of int or str, optional): Class IDs or names considered as pollutants.
                                                            If None, all classes detected are considered relevant.
        """
        self.detector = yolo_detector
        self.relevance_threshold = relevance_threshold if relevance_threshold is not None else self.detector.conf_thresh
        
        self.relevant_class_ids = None
        if relevant_classes:
            self.relevant_class_ids = set()
            model_class_names = self.detector.get_names()
            for cls in relevant_classes:
                if isinstance(cls, str):
                    try:
                        cls_id = model_class_names.index(cls)
                        self.relevant_class_ids.add(cls_id)
                    except ValueError:
                        logger.warning(f"Class name '{cls}' not found in detector's class list: {model_class_names}")
                elif isinstance(cls, int):
                    if 0 <= cls < len(model_class_names):
                        self.relevant_class_ids.add(cls)
                    else:
                        logger.warning(f"Class ID {cls} is out of range for detector's classes (0-{len(model_class_names)-1}).")
            if not self.relevant_class_ids:
                logger.warning("No valid relevant classes specified or found. Filter will consider any detection.")
                self.relevant_class_ids = None # Fallback to any detection
        
        logger.info(f"FrameFilter initialized. Relevance threshold: {self.relevance_threshold}, Relevant class IDs: {self.relevant_class_ids or 'Any'}")


    def is_frame_relevant(self, frame_bgr):
        """
        Determines if a frame is relevant by checking for pollutant detections.

        Args:
            frame_bgr (np.array): The image frame in BGR format.

        Returns:
            tuple: (bool, list_of_detections)
                   True if relevant (pollutants detected), False otherwise.
                   The list contains detections that made the frame relevant.
        """
        detections = self.detector.detect(frame_bgr)
        relevant_detections_in_frame = []

        if not detections:
            return False, []

        for det in detections:
            if det['confidence'] >= self.relevance_threshold:
                if self.relevant_class_ids is None: # Any detection above threshold is relevant
                    relevant_detections_in_frame.append(det)
                elif det['class_id'] in self.relevant_class_ids:
                    relevant_detections_in_frame.append(det)
        
        return len(relevant_detections_in_frame) > 0, relevant_detections_in_frame

if __name__ == '__main__':
    from models.yolov5_lightweight.detect_yolov5 import YOLOv5Detector # Assuming this is implemented
    import cv2

    try:
        # Initialize the YOLOv5 detector (ensure weights are available)
        yolo_detector_instance = YOLOv5Detector(conf_thresh=0.3) # Lower threshold for filtering maybe
        
        # Define which classes indicate pollutants (use names from your trained YOLO model)
        # Example: pollutant_class_names = ['algae', 'trash', 'plastic_bottle']
        pollutant_class_names_config = config.get("inference.filter_relevant_classes", ["algae", "trash"]) # From config
        
        frame_fltr = PollutantFrameFilter(yolo_detector_instance, relevant_classes=pollutant_class_names_config)

        # Create a dummy image or load one
        dummy_img_path_filter = "temp_filter_test.jpg" # Use an image that might have detections
        if not os.path.exists(dummy_img_path_filter):
             # Try to use the one from YOLO detector test
            dummy_img_path_filter = "temp_yolo_detector_test.jpg" 
            if not os.path.exists(dummy_img_path_filter):
                logger.warning(f"Test image {dummy_img_path_filter} not found. Frame filter test will be limited.")
                frame_to_test = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) # Random noise
            else:
                frame_to_test = cv2.imread(dummy_img_path_filter)
        else:
            frame_to_test = cv2.imread(dummy_img_path_filter)

        if frame_to_test is not None:
            logger.info("Testing frame filter...")
            is_relevant, relevant_dets = frame_fltr.is_frame_relevant(frame_to_test)
            if is_relevant:
                logger.info(f"Frame IS relevant. Detections ({len(relevant_dets)}):")
                for d in relevant_dets:
                    logger.info(f"  - {d['class_name']} at {d['bbox']} (conf: {d['confidence']:.2f})")
            else:
                logger.info("Frame is NOT relevant (no pollutants detected above threshold).")
        else:
            logger.error("Could not load or create a frame for testing the filter.")

    except FileNotFoundError as e:
        logger.error(f"YOLOv5 weights file not found for FrameFilter test: {e}")
    except Exception as e:
        logger.error(f"Error in PollutantFrameFilter example: {e}")
        import traceback
        traceback.print_exc()