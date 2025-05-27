import os
import sys
import subprocess
import torch
from utils.config_parser import ConfigParser
from utils.logger import setup_logger

# This script assumes you have the Ultralytics YOLOv5 repository
# cloned and accessible, or that 'yolov5' is installed as a package
# that provides a training entry point.

# Path to the Ultralytics YOLOv5 train.py script
# Option 1: If yolov5 is a cloned repo adjacent to your project
YOLOV5_TRAIN_SCRIPT_PATH_REPO = "../yolov5/train.py" # Adjust if your yolov5 repo is elsewhere
# Option 2: If yolov5 is installed as a package (less common for direct train.py access)
# No direct path, you'd call a module: python -m yolov5.train ...

logger = setup_logger("yolov5_lightweight_trainer")
config = ConfigParser()

def check_yolov5_train_script(script_path):
    if not os.path.exists(script_path):
        logger.error(f"YOLOv5 train.py not found at {script_path}.")
        logger.error("Please ensure the Ultralytics YOLOv5 repository is cloned and the path is correct,")
        logger.error("or that YOLOv5 is installed in a way that provides a training script.")
        return False
    return True

def train_yolov5_lightweight():
    logger.info("Starting training for Lightweight YOLOv5 model.")

    # Configuration from config.yaml
    img_size_train = config.get("yolov5_params.img_size_train", [config.get('image_width', 1280), config.get('image_height', 720)]) # [width, height]
    if isinstance(img_size_train, list) and len(img_size_train) == 2: # ultralytics train.py expects single int for square, or WxH
        img_size_train_arg = max(img_size_train) # Use max for --imgsz if not providing two values
        # Or potentially pass both if your version of train.py supports it. Common is single value for largest dim.
        # The paper used 1280x720. Let's pass 1280.
        img_size_train_arg = config.get('image_width', 1280)
    else:
        img_size_train_arg = 640 # default fallback


    batch_size = config.get("batch_size", 16)
    epochs = config.get("epochs_yolo", 50)
    # For learning rates, the paper uses SGD. YOLOv5 train.py defaults to SGD.
    # It has --hyp for hyperparameters YAML, which includes lr0, lrf.
    # Initial LR (lr0) 0.01, Final LR (lrf via lr0*lrf) 0.001 (lrf=0.1)
    # For simplicity, we can let train.py use its default hyperparameter evolution or a base hyp.yaml.
    # The paper mentions SGD initial 0.01, Adam initial 0.001, Final 0.001.
    # YOLOv5's train.py uses SGD by default. Can specify optimizer with --optimizer Adam
    
    custom_model_config = os.path.abspath(config.get("yolov5_config_path")) # e.g., models/yolov5_lightweight/model.yaml
    dataset_config = os.path.abspath(config.get("yolov5_dataset_yaml"))   # e.g., config/dataset_yolo.yaml
    
    # Base weights: start from yolov5s.pt or a pretrained checkpoint
    # You need to download yolov5s.pt if you don't have it
    base_weights = config.get("yolov5_params.base_weights", "yolov5s.pt") 
    if not os.path.exists(base_weights):
        logger.warning(f"Base weights {base_weights} not found. Training will be from scratch if not specified otherwise in yolov5 train.py.")
        # Consider downloading it automatically here if you want
        # For example: torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

    project_name = "uav_water_pollutant_runs"
    experiment_name = config.get("yolov5_params.experiment_name", "lightweight_yolov5_custom_run")

    # Device: '' for auto (CPU/GPU), or '0' for GPU 0, 'cpu' for CPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Construct the command
    # Check if yolov5 repo path exists for direct script call
    yolov5_repo_root = None
    potential_paths = ["../yolov5", "yolov5", "../../yolov5"] # Common relative paths
    for p_path in potential_paths:
        if os.path.isdir(p_path) and os.path.exists(os.path.join(p_path, "train.py")):
            yolov5_repo_root = os.path.abspath(p_path)
            break
    
    if not yolov5_repo_root:
        logger.error("Could not automatically find YOLOv5 repository root.")
        logger.error("Please ensure yolov5 is cloned, e.g., adjacent to this project directory or install it.")
        logger.info("Attempting to run `python -m yolov5.train` if installed as package.")
        cmd_prefix = [sys.executable, "-m", "yolov5.train"]

    else:
        train_script = os.path.join(yolov5_repo_root, "train.py")
        if not check_yolov5_train_script(train_script):
            return
        cmd_prefix = [sys.executable, train_script]


    command = cmd_prefix + [
        "--imgsz", str(img_size_train_arg),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--data", dataset_config,
        "--cfg", custom_model_config,
        "--weights", base_weights if os.path.exists(base_weights) else "", # Pass empty if not found, train.py handles it
        "--project", project_name,
        "--name", experiment_name,
        "--device", device,
        # Add other YOLOv5 training arguments as needed from config or paper
        # E.g., --hyp path/to/hyp.yaml if you have custom hyperparameters
        # The paper mentions optimizer details, which train.py handles via hyp.yaml or defaults.
        # If you need to force SGD with specific LR:
        # This might require modifying train.py or using a hyp file.
        # Standard train.py uses SGD with lr0=0.01, lrf=0.01 (newer versions) or lrf=0.1 (older versions).
        # To match paper SGD initial 0.01, final 0.001: lr0=0.01, lrf=0.1
        # Add: "--hyp", "path_to_hyp_with_lr0_0.01_lrf_0.1.yaml"
        # Or if train.py supports it: "--lr0", "0.01", "--lrf", "0.1" (check train.py --help)
        "--nosave", # To prevent saving checkpoints every epoch unless specified by --save-period
        "--save-period", "10", # Save every 10 epochs
        "--patience", "20" # Early stopping patience
    ]

    logger.info(f"Executing YOLOv5 training command: {' '.join(command)}")

    try:
        # Set PYTHONPATH if calling script from a different directory structure,
        # to ensure yolov5 modules are found if train.py is in a subdirectory.
        env = os.environ.copy()
        if yolov5_repo_root and cmd_prefix[1] != "-m": # If calling train.py directly
             env["PYTHONPATH"] = f"{yolov5_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line) # Print YOLOv5 output in real-time
            logger.debug(line.strip()) # Also log it
        process.wait()
        if process.returncode == 0:
            logger.info("YOLOv5 training completed successfully.")
            # The trained weights will be in project_name/experiment_name/weights/best.pt or last.pt
            # Update config.yolov5_weights_path if needed
            trained_weights_path = os.path.join(project_name, experiment_name, "weights", "best.pt")
            if os.path.exists(trained_weights_path):
                 logger.info(f"Best trained weights saved at: {os.path.abspath(trained_weights_path)}")
                 # You might want to copy 'best.pt' to config.get('yolov5_weights_path')
                 shutil.copy2(trained_weights_path, config.get('yolov5_weights_path'))
                 logger.info(f"Copied best weights to: {config.get('yolov5_weights_path')}")

            else:
                logger.warning(f"Could not find best.pt at expected location: {trained_weights_path}")
        else:
            logger.error(f"YOLOv5 training failed with return code {process.returncode}.")

    except FileNotFoundError:
        logger.error("Error: train.py or python interpreter not found. Ensure paths are correct and Python is installed.")
    except Exception as e:
        logger.error(f"An error occurred during YOLOv5 training: {e}")

if __name__ == "__main__":
    # Before running:
    # 1. Ensure `config/dataset_yolo.yaml` is correctly set up.
    # 2. Ensure `models/yolov5_lightweight/model.yaml` defines your custom architecture.
    # 3. Download `yolov5s.pt` (or other base weights) or let training start from scratch.
    # 4. Make sure you have the Ultralytics YOLOv5 repository cloned or installed.
    #    If cloned, adjust YOLOV5_TRAIN_SCRIPT_PATH_REPO or the auto-detection logic.
    import shutil # Add import for shutil
    train_yolov5_lightweight()