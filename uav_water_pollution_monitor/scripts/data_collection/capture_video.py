import cv2
import time
import os
from datetime import datetime
from utils.logger import setup_logger

# This script is conceptual. Actual implementation depends on your UAV's camera SDK or interface.
# For a simple webcam connected to the R-Pi (if R-Pi is also the flight controller or connected to it):
# For a DJI drone or similar, you'd use their SDK (e.g., DJITelloPy, DJI SDK).

logger = setup_logger("video_capture")

def capture_video_from_source(source_id=0, output_dir="data/raw/video", duration_sec=300, fps=30, width=1280, height=720):
    """
    Captures video from a specified camera source.

    Args:
        source_id (int or str): Camera index (e.g., 0 for webcam) or video stream URL.
        output_dir (str): Directory to save the captured video.
        duration_sec (int): Duration of the video capture in seconds.
        fps (int): Frames per second for the output video.
        width (int): Frame width.
        height (int): Frame height.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(source_id)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video source {source_id}")
        return

    # Set properties (may not work for all cameras/sources)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Attempting to capture at {width}x{height} @ {fps} FPS.")
    logger.info(f"Actual camera resolution: {actual_width}x{actual_height} @ {actual_fps if actual_fps > 0 else 'N/A'} FPS.")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"uav_capture_{timestamp}.mp4") # Or .avi

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') # For .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For .mp4
    out = cv2.VideoWriter(output_filename, fourcc, fps, (actual_width, actual_height))

    start_time = time.time()
    frames_written = 0
    logger.info(f"Starting video capture for {duration_sec} seconds. Output: {output_filename}")

    try:
        while (time.time() - start_time) < duration_sec:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                frames_written += 1
                # cv2.imshow('Capturing...', frame) # Optional: display the frame
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     logger.info("Capture interrupted by user.")
                #     break
            else:
                logger.warning("Failed to grab frame.")
                break
    except KeyboardInterrupt:
        logger.info("Capture interrupted by user (KeyboardInterrupt).")
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Video capture finished. Duration: {elapsed_time:.2f}s. Frames written: {frames_written}")
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    VIDEO_SOURCE = 0  # 0 for default webcam, or path to video file/stream URL
                      # For UAV, this would be specific to its camera access method
    OUTPUT_DIRECTORY = "data/raw/video_captures"
    CAPTURE_DURATION_SECONDS = 10 # Short duration for testing
    TARGET_FPS = 20
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720
    # --- END CONFIGURATION ---

    logger.info("Attempting to start video capture...")
    capture_video_from_source(
        source_id=VIDEO_SOURCE,
        output_dir=OUTPUT_DIRECTORY,
        duration_sec=CAPTURE_DURATION_SECONDS,
        fps=TARGET_FPS,
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT
    )
    logger.info("Video capture script finished.")