import pandas as pd
import numpy as np
import os
from utils.config_parser import ConfigParser
from utils.logger import setup_logger

logger = setup_logger("sensor_loader")
config = ConfigParser()

class SensorFeatureLoader:
    def __init__(self, sensor_data_file=None, relevant_features=None):
        self.sensor_data_file = sensor_data_file if sensor_data_file else config.get("sensor_data_file")
        self.relevant_features = relevant_features if relevant_features else config.get("sensor_features", ["ph", "tds"])
        
        if not self.sensor_data_file or not os.path.exists(self.sensor_data_file):
            logger.warning(f"Sensor data file not found: {self.sensor_data_file}. Sensor features will be unavailable or dummy.")
            self.data = None
        else:
            try:
                self.data = pd.read_csv(self.sensor_data_file)
                logger.info(f"Loaded sensor data from: {self.sensor_data_file} with columns: {self.data.columns.tolist()}")
                # Basic preprocessing: handle missing values (e.g., fill with mean or a specific value)
                for col in self.relevant_features:
                    if col in self.data.columns:
                        if self.data[col].isnull().any():
                            mean_val = self.data[col].mean()
                            self.data[col].fillna(mean_val, inplace=True)
                            logger.info(f"Filled NaNs in sensor column '{col}' with mean value: {mean_val:.2f}")
                    else:
                        logger.warning(f"Relevant sensor feature '{col}' not found in sensor data file. Will use NaN/0 if requested.")
                        # Add a dummy column with NaNs if it doesn't exist so get_features doesn't fail
                        self.data[col] = np.nan


            except Exception as e:
                logger.error(f"Error loading or processing sensor data file {self.sensor_data_file}: {e}")
                self.data = None
        
        self.num_features = len(self.relevant_features)

    def get_features(self, image_filename):
        """
        Retrieves sensor features corresponding to an image filename.
        Assumes 'image_filename' column exists in the sensor_data_csv.

        Args:
            image_filename (str): The base name of the image file (e.g., "frame_0001.jpg").

        Returns:
            np.array: Array of sensor features, or an array of NaNs/zeros if not found or error.
        """
        if self.data is None:
            logger.debug(f"No sensor data loaded, returning dummy features for {image_filename}")
            return np.full(self.num_features, np.nan) # Or np.zeros(self.num_features)

        try:
            # Match by base filename
            base_img_name = os.path.basename(image_filename)
            row = self.data[self.data['image_filename'] == base_img_name]
            
            if not row.empty:
                # Ensure all relevant features are present, use NaN if a column was missing during load
                features = [row.iloc[0].get(feat, np.nan) for feat in self.relevant_features]
                return np.array(features, dtype=np.float32)
            else:
                logger.debug(f"No sensor data found for image: {base_img_name}. Returning NaNs.")
                return np.full(self.num_features, np.nan)
        except Exception as e:
            logger.error(f"Error retrieving sensor features for {image_filename}: {e}")
            return np.full(self.num_features, np.nan)

    def get_feature_names(self):
        return self.relevant_features

    def get_num_features(self):
        return self.num_features

if __name__ == '__main__':
    # Create a dummy sensor_features.csv for testing
    dummy_csv_path = "dummy_sensor_data.csv"
    if not os.path.exists(dummy_csv_path):
        dummy_data = {
            'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:00:05', '2023-01-01 10:00:10', '2023-01-01 10:00:15']),
            'image_filename': ['frame_0001.jpg', 'frame_0002.jpg', 'frame_0003.jpg', 'frame_0004.jpg'],
            'ph': [7.1, 7.2, None, 7.0], # Added a None to test NaN handling
            'tds': [350.5, 352.0, 351.2, 349.8],
            'temperature': [25.1, 25.2, 25.1, 25.0] # Extra feature not in default relevant_features
        }
        pd.DataFrame(dummy_data).to_csv(dummy_csv_path, index=False)
        logger.info(f"Created dummy sensor data file: {dummy_csv_path}")

    # Test with default config (which might point to a non-existent file initially)
    logger.info("--- Testing with default config path ---")
    sfl_default = SensorFeatureLoader()
    print(f"Default relevant features: {sfl_default.get_feature_names()}")
    print(f"Default features for 'frame_0001.jpg': {sfl_default.get_features('frame_0001.jpg')}")


    logger.info("\n--- Testing with dummy_sensor_data.csv ---")
    # Override relevant_features for testing
    relevant_feats_test = ["ph", "tds", "turbidity"] # turbidity is not in dummy csv
    sfl = SensorFeatureLoader(sensor_data_file=dummy_csv_path, relevant_features=relevant_feats_test)
    
    print(f"Feature names: {sfl.get_feature_names()}")
    print(f"Number of features: {sfl.get_num_features()}")

    features_img1 = sfl.get_features('frame_0001.jpg')
    print(f"Features for frame_0001.jpg: {features_img1}") # Expected: [7.1, 350.5, nan]

    features_img3 = sfl.get_features('frame_0003.jpg') # ph is None in CSV, should be mean
    print(f"Features for frame_0003.jpg: {features_img3}") # Expected: [mean_ph, 351.2, nan]
    
    features_img_nonexist = sfl.get_features('frame_9999.jpg')
    print(f"Features for frame_9999.jpg (non-existent): {features_img_nonexist}") # Expected: [nan, nan, nan]

    # Clean up dummy file
    # if os.path.exists(dummy_csv_path):
    #     os.remove(dummy_csv_path)