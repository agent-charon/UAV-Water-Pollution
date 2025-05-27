import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.config_parser import ConfigParser
from utils.logger import setup_logger

logger = setup_logger("vit_extractor")
config = ConfigParser()

class ViTExtractor:
    def __init__(self, model_name=None, checkpoint_path=None, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"ViT Extractor using device: {self.device}")

        self.model_name = model_name if model_name else config.get("vit_model_name", "vit_base_patch16_224")
        
        try:
            # Load a pretrained ViT model from timm
            # We want to extract features, so we'll remove the classification head
            self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0) # num_classes=0 removes head
            
            # Load custom checkpoint if provided (e.g., if ViT was fine-tuned)
            _checkpoint_path = checkpoint_path if checkpoint_path else config.get("vit_checkpoint_path")
            if _checkpoint_path and os.path.exists(_checkpoint_path):
                logger.info(f"Loading ViT checkpoint from: {_checkpoint_path}")
                self.model.load_state_dict(torch.load(_checkpoint_path, map_location=self.device))
            elif _checkpoint_path:
                logger.warning(f"ViT checkpoint path specified but not found: {_checkpoint_path}. Using pretrained weights.")

            self.model.eval()
            self.model.to(self.device)
            
            # Get the input size and preprocessing transforms expected by the model
            data_config = timm.data.resolve_data_config(self.model.pretrained_cfg if hasattr(self.model, 'pretrained_cfg') else self.model.default_cfg)
            self.transform = timm.data.create_transform(**data_config, is_training=False)
            self.input_size = data_config['input_size'] # (C, H, W)
            logger.info(f"ViT Model: {self.model_name} loaded. Expected input size: {self.input_size}")
            logger.info(f"ViT Transform: {self.transform}")

        except Exception as e:
            logger.error(f"Error initializing ViT model {self.model_name}: {e}")
            raise

    def extract_features(self, image_path_or_pil):
        """
        Extracts features from a single image.

        Args:
            image_path_or_pil (str or PIL.Image): Path to the image or a PIL Image object.

        Returns:
            np.array: Extracted feature vector, or None if an error occurs.
        """
        try:
            if isinstance(image_path_or_pil, str):
                if not os.path.exists(image_path_or_pil):
                    logger.error(f"Image not found for ViT extraction: {image_path_or_pil}")
                    return None
                img = Image.open(image_path_or_pil).convert('RGB')
            elif isinstance(image_path_or_pil, Image.Image):
                img = image_path_or_pil.convert('RGB')
            elif isinstance(image_path_or_pil, np.ndarray): # if image is np array (H,W,C)
                img = Image.fromarray(image_path_or_pil).convert('RGB')
            else:
                logger.error("Invalid input type for ViT extraction. Must be path, PIL Image, or NumPy array.")
                return None

            img_tensor = self.transform(img).unsqueeze(0).to(self.device) # Add batch dimension

            with torch.no_grad():
                features = self.model(img_tensor) # (batch_size, feature_dim)
            
            return features.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(f"Error during ViT feature extraction: {e}")
            return None

    def get_feature_dimension(self):
        """
        Returns the output feature dimension of the ViT model.
        This might need to be explicitly known or inferred.
        For timm models with num_classes=0, the output is the feature vector before the head.
        """
        # Try to infer from model config or use a known value from config.yaml
        default_dim = config.get("vit_feature_dim")
        try:
            # Create a dummy input
            dummy_input = torch.randn(1, self.input_size[0], self.input_size[1], self.input_size[2]).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input)
            return output.shape[1]
        except Exception as e:
            logger.warning(f"Could not dynamically determine ViT feature dimension: {e}. Using default from config: {default_dim}")
            return default_dim


if __name__ == '__main__':
    import os
    # Create a dummy image for testing
    try:
        from PIL import Image, ImageDraw
        dummy_image_path = "dummy_vit_test_image.png"
        if not os.path.exists(dummy_image_path):
            img = Image.new('RGB', (224, 224), color = 'red')
            draw = ImageDraw.Draw(img)
            draw.rectangle(((50,50),(150,150)), fill="blue")
            img.save(dummy_image_path)
            logger.info(f"Created dummy image: {dummy_image_path}")

        extractor = ViTExtractor()
        logger.info(f"ViT Feature Dimension: {extractor.get_feature_dimension()}")
        
        if os.path.exists(dummy_image_path):
            features = extractor.extract_features(dummy_image_path)
            if features is not None:
                logger.info(f"Extracted features shape: {features.shape}")
                logger.info(f"First 5 features: {features[:5]}")
            else:
                logger.error("Failed to extract features from dummy image.")
            # os.remove(dummy_image_path) # Clean up
        else:
            logger.error(f"Dummy image {dummy_image_path} not found for testing.")

    except ImportError:
        logger.error("Pillow (PIL) is not installed. Cannot run ViT Extractor example.")
    except Exception as e:
        logger.error(f"Error in ViT Extractor example: {e}")