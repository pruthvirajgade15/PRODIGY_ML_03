import os
import sys
import pickle
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj) -> None:
    """Save a Python object to a pickle file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at: {file_path}")
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Load a Python object from a pickle file."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


def load_and_preprocess_image(image_path: str, target_size: tuple = (64, 64)) -> np.ndarray:
    """
    Read an image from disk, resize it, and convert to grayscale.
    
    Args:
        image_path: Path to the image file
        target_size: Target dimensions (width, height)
    
    Returns:
        Grayscale resized image as numpy array
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image = cv2.resize(image, target_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    
    except Exception as e:
        raise CustomException(e, sys)


def preprocess_image_from_array(image_array: np.ndarray, target_size: tuple = (64, 64)) -> np.ndarray:
    """
    Preprocess an image from a numpy array (e.g., from uploaded file).
    
    Args:
        image_array: Image as numpy array (BGR or RGB)
        target_size: Target dimensions (width, height)
    
    Returns:
        Grayscale resized image as numpy array
    """
    try:
        image = cv2.resize(image_array, target_size)
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return gray
    
    except Exception as e:
        raise CustomException(e, sys)


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from a grayscale image.
    
    Args:
        image: Grayscale image as numpy array
    
    Returns:
        HOG feature vector
    """
    try:
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        return features
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained sklearn model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary with accuracy, classification_report, and confusion_matrix
    """
    try:
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'], output_dict=True)
        report_text = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'])
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'classification_report_text': report_text,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        logging.info(f"Model Accuracy: {accuracy:.4f}")
        logging.info(f"\nClassification Report:\n{report_text}")
        
        return results
    
    except Exception as e:
        raise CustomException(e, sys)
