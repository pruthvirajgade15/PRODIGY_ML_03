import os
import sys
import numpy as np
import cv2
from PIL import Image

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    load_object,
    extract_hog_features,
    preprocess_image_from_array
)


class CustomData:
    """
    Handles custom input data for prediction.
    
    Accepts images from various sources:
    - File path (string)
    - PIL Image object (from Streamlit upload)
    - Numpy array (from OpenCV)
    """
    
    def __init__(self, image=None, image_path: str = None):
        self.image = image
        self.image_path = image_path
    
    def get_image_array(self) -> np.ndarray:
        """Convert input to numpy array (BGR format for OpenCV)."""
        try:
            if self.image_path is not None:
                # Read from file path
                img = cv2.imread(self.image_path)
                if img is None:
                    raise ValueError(f"Could not read image: {self.image_path}")
                return img
            
            elif self.image is not None:
                if isinstance(self.image, np.ndarray):
                    return self.image
                
                elif isinstance(self.image, Image.Image):
                    # Convert PIL Image to numpy array (RGB -> BGR)
                    img_array = np.array(self.image)
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    return img_array
                
                else:
                    raise ValueError(f"Unsupported image type: {type(self.image)}")
            
            else:
                raise ValueError("No image provided. Provide either 'image' or 'image_path'.")
        
        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    """
    Prediction pipeline for classifying cat/dog images.
    
    Loads the trained SVM model and preprocessor,
    processes input images, and returns predictions with confidence.
    """
    
    CLASS_NAMES = {0: 'Cat', 1: 'Dog'}
    
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model = None
        self.preprocessor = None
    
    def _load_artifacts(self):
        """Load model and preprocessor from disk."""
        try:
            if self.model is None:
                logging.info("Loading model and preprocessor...")
                self.model = load_object(self.model_path)
                self.preprocessor = load_object(self.preprocessor_path)
                logging.info("Artifacts loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(self, custom_data: CustomData) -> dict:
        """
        Predict whether an image contains a cat or a dog.
        
        Args:
            custom_data: CustomData object containing the image
        
        Returns:
            Dictionary with:
            - prediction: 'Cat' or 'Dog'
            - confidence: float (0-1)
            - probabilities: dict with class probabilities
        """
        try:
            self._load_artifacts()
            
            # Get image array
            img_array = custom_data.get_image_array()
            
            # Preprocess: resize and convert to grayscale
            gray_image = preprocess_image_from_array(img_array, target_size=(64, 64))
            
            # Extract HOG features
            hog_features = extract_hog_features(gray_image)
            
            # Reshape for single prediction
            features = hog_features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.preprocessor.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            
            # Get probability scores
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            result = {
                'prediction': self.CLASS_NAMES[prediction],
                'label': int(prediction),
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'Cat': float(probabilities[0]),
                    'Dog': float(probabilities[1])
                }
            }
            
            logging.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
            
            return result
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def is_model_available(self) -> bool:
        """Check if trained model artifacts exist."""
        return (
            os.path.exists(self.model_path) and
            os.path.exists(self.preprocessor_path)
        )
