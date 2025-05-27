import logging
import os
from datetime import datetime

def setup_logger(name="uav_water_detector", level=logging.INFO, log_dir="logs"):
    """
    Sets up a logger.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger

if __name__ == '__main__':
    # Example Usage
    logger = setup_logger(name="my_app_logger")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")

    another_logger = setup_logger(name="another_module_logger", level=logging.DEBUG)
    another_logger.debug("This is a debug message for another module.")