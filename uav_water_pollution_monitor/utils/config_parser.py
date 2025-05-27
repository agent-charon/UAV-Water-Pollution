import yaml
import os

class ConfigParser:
    def __init__(self, config_path="config/config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        """
        Retrieves a configuration value for a given key.
        Supports nested keys using dot notation (e.g., "yolov5_params.batch_size").
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else: # handle cases where a sub-key is requested but parent is not a dict
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key):
        """Allows dictionary-style access."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Key '{key}' not found in configuration.")
        return value

    def get_all(self):
        return self.config

if __name__ == '__main__':
    # Example usage
    try:
        config = ConfigParser()
        print(f"Data directory: {config.get('data_dir')}")
        print(f"YOLOv5 Batch Size: {config.get('training_params.batch_size_yolo')}") # Example of nested key
        print(f"Non-existent key with default: {config.get('non_existent_key', 'default_value')}")
        print(f"All Config: {config.get_all()}")
        print(f"Access via __getitem__: {config['models_dir']}")
    except FileNotFoundError as e:
        print(e)
    except KeyError as e:
        print(e)