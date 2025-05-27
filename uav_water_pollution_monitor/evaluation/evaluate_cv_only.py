import os
import sys
import subprocess
import torch
from utils.config_parser import ConfigParser
from utils.logger import setup_logger
import glob
import shutil

# This script is for evaluating the Computer Vision (CV) only model,
# which is the lightweight YOLOv5 model in this paper.
# It will use the Ultralytics YOLOv5 val.py (or detect.py in validation mode)
# to get mAP, precision, recall on a test set.

logger = setup_logger("cv_only_evaluator")
config = ConfigParser()

def check_yolov5_val_script(script_path):
    if not os.path.exists(script_path):
        logger.error(f"YOLOv5 val.py (or detect.py) not found at {script_path}.")
        logger.error("Please ensure the Ultralytics YOLOv5 repository is cloned and the path is correct,")
        logger.error("or that YOLOv5 is installed in a way that provides an evaluation script.")
        return False
    return True

def evaluate_lightweight_yolov5():
    logger.info("Starting evaluation for Lightweight YOLOv5 model (CV only).")

    # Configuration from config.yaml
    # Use image dimensions specified for the dataset, paper uses 1280x720
    img_size_eval_list = [config.get('image_width', 1280), config.get('image_height', 720)]
    img_size_eval = max(img_size_eval_list) # val.py usually takes a single int for image size

    batch_size = config.get("evaluation.cv_batch_size", config.get("batch_size", 16)) # Use a specific eval batch size or training one
    
    # Path to the trained lightweight YOLOv5 model weights
    trained_weights = config.get("yolov5_weights_path")
    if not trained_weights or not os.path.exists(trained_weights):
        logger.error(f"Trained YOLOv5 weights not found at: {trained_weights}")
        logger.error("Please train the model first or provide the correct path in config.yaml.")
        # Try to find the latest run if not specified
        latest_run_dir = os.path.join("uav_water_pollutant_runs", config.get("yolov5_params.experiment_name", "lightweight_yolov5_custom_run"))
        potential_weights_path = os.path.join(latest_run_dir, "weights", "best.pt")
        if os.path.exists(potential_weights_path):
            logger.info(f"Found potential weights at {potential_weights_path}, using this.")
            trained_weights = potential_weights_path
        else:
            logger.error(f"Could not find weights in default run location either: {potential_weights_path}")
            return

    dataset_config = os.path.abspath(config.get("yolov5_dataset_yaml")) # This YAML should define train/val/test paths
    
    # Device: '' for auto (CPU/GPU), or '0' for GPU 0, 'cpu' for CPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device} for evaluation.")

    project_name = "uav_water_pollutant_evaluations"
    eval_name = config.get("evaluation.cv_eval_name", "lightweight_yolov5_eval")

    # Determine path to val.py or detect.py
    yolov5_repo_root = None
    script_to_use = "val.py" # Newer YOLOv5 uses val.py, older might use test.py or detect.py with --task val

    potential_paths = ["../yolov5", "yolov5", "../../yolov5"]
    for p_path in potential_paths:
        if os.path.isdir(p_path) and os.path.exists(os.path.join(p_path, script_to_use)):
            yolov5_repo_root = os.path.abspath(p_path)
            break
    
    if not yolov5_repo_root: # Try detect.py if val.py not found
        script_to_use = "detect.py" 
        for p_path in potential_paths:
            if os.path.isdir(p_path) and os.path.exists(os.path.join(p_path, script_to_use)):
                yolov5_repo_root = os.path.abspath(p_path)
                break
    
    cmd_prefix = []
    if not yolov5_repo_root:
        logger.error(f"Could not automatically find YOLOv5 repository root with {script_to_use}.")
        logger.info(f"Attempting to run `python -m yolov5.{script_to_use.split('.')[0]}` if installed as package.")
        cmd_prefix = [sys.executable, "-m", f"yolov5.{script_to_use.split('.')[0]}"]
    else:
        eval_script_path = os.path.join(yolov5_repo_root, script_to_use)
        if not check_yolov5_val_script(eval_script_path):
            return
        cmd_prefix = [sys.executable, eval_script_path]

    # Command arguments
    command_args = [
        "--data", dataset_config,
        "--weights", trained_weights,
        "--imgsz", str(img_size_eval),
        "--batch-size", str(batch_size),
        "--device", device,
        "--project", project_name,
        "--name", eval_name,
        "--verbose", # Get detailed output
        "--save-txt", # Save results to text files
        "--save-hybrid", # Save hybrid labels (txt + conf)
        "--save-conf" # Save confidences in --save-txt files
    ]
    if script_to_use == "val.py" or (script_to_use == "detect.py" and "--task" in subprocess.getoutput(f"{' '.join(cmd_prefix)} --help")):
         command_args.extend(["--task", "val"]) # Ensure validation task for val.py or newer detect.py

    command = cmd_prefix + command_args
    logger.info(f"Executing YOLOv5 evaluation command: {' '.join(command)}")

    try:
        env = os.environ.copy()
        if yolov5_repo_root and cmd_prefix[1] != "-m":
             env["PYTHONPATH"] = f"{yolov5_repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            logger.debug(line.strip())
        process.wait()

        if process.returncode == 0:
            logger.info("YOLOv5 evaluation completed successfully.")
            results_path = os.path.join(project_name, eval_name)
            logger.info(f"Evaluation results, plots, and logs should be in: {os.path.abspath(results_path)}")
            # You can parse the output logs or specific result files (e.g., results.csv if generated)
            # to extract mAP, P, R values programmatically.
            # For example, val.py typically prints these to stdout and saves them.
        else:
            logger.error(f"YOLOv5 evaluation failed with return code {process.returncode}.")

    except FileNotFoundError:
        logger.error(f"Error: {script_to_use} or python interpreter not found. Ensure paths are correct.")
    except Exception as e:
        logger.error(f"An error occurred during YOLOv5 evaluation: {e}")

if __name__ == "__main__":
    # Before running:
    # 1. Ensure `config/dataset_yolo.yaml` points to your dataset, especially the 'test' split path if specified,
    #    otherwise val.py will use the 'val' split for evaluation.
    # 2. Ensure `models_dir/yolov5_lightweight.pt` (or the path from config.yolov5_weights_path) exists
    #    and contains your trained lightweight YOLOv5 model.
    # 3. Make sure Ultralytics YOLOv5 repo is accessible.
    evaluate_lightweight_yolov5()