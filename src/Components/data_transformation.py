import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    save_object,
    load_and_preprocess_image,
    extract_hog_features
)


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation."""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    image_size: tuple = (64, 64)
    max_samples_per_class: int = 2000  # Limit for reasonable SVM training time


class DataTransformation:
    """
    Handles image preprocessing and HOG feature extraction.
    
    Pipeline:
    1. Read images from organized directories
    2. Resize to target dimensions (64x64)
    3. Convert to grayscale
    4. Extract HOG features
    5. Normalize with StandardScaler
    """
    
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def extract_features_from_directory(self, directory: str, max_samples: int = None) -> tuple:
        """
        Extract HOG features from all images in a directory structure.
        
        Expected directory structure:
        directory/
        ├── cats/
        │   ├── cat.0.jpg
        │   └── ...
        └── dogs/
            ├── dog.0.jpg
            └── ...
        
        Args:
            directory: Path to directory containing 'cats' and 'dogs' subdirectories
            max_samples: Maximum number of samples per class
        
        Returns:
            Tuple of (features_array, labels_array)
        """
        try:
            if max_samples is None:
                max_samples = self.transformation_config.max_samples_per_class
            
            features = []
            labels = []
            
            categories = {'cats': 0, 'dogs': 1}
            
            for category, label in categories.items():
                category_path = os.path.join(directory, category)
                
                if not os.path.exists(category_path):
                    logging.warning(f"Category directory not found: {category_path}")
                    continue
                
                image_files = [
                    f for f in os.listdir(category_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                
                # Limit samples
                image_files = image_files[:max_samples]
                
                processed = 0
                skipped = 0
                
                for img_file in image_files:
                    try:
                        img_path = os.path.join(category_path, img_file)
                        
                        # Load and preprocess image
                        gray_image = load_and_preprocess_image(
                            img_path,
                            self.transformation_config.image_size
                        )
                        
                        # Extract HOG features
                        hog_features = extract_hog_features(gray_image)
                        
                        features.append(hog_features)
                        labels.append(label)
                        processed += 1
                    
                    except Exception as img_error:
                        skipped += 1
                        continue
                
                logging.info(
                    f"  {category}: processed={processed}, skipped={skipped}"
                )
                print(f"      {category}: {processed} processed, {skipped} skipped")
            
            return np.array(features), np.array(labels)
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple:
        """
        Complete data transformation pipeline.
        
        Args:
            train_path: Path to training data directory
            test_path: Path to test data directory
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, preprocessor_path)
        """
        try:
            logging.info(">>> Data Transformation Started <<<")
            print("\n" + "=" * 60)
            print("🔄 STEP 2: Data Transformation (HOG Feature Extraction)")
            print("=" * 60)
            
            # Extract features from training images
            logging.info("Extracting features from training images...")
            print("   Extracting HOG features from TRAINING images...")
            X_train, y_train = self.extract_features_from_directory(train_path)
            logging.info(f"Training features shape: {X_train.shape}")
            print(f"   ✅ Training features shape: {X_train.shape}")
            
            # Extract features from test images
            logging.info("Extracting features from test images...")
            print("   Extracting HOG features from TEST images...")
            X_test, y_test = self.extract_features_from_directory(test_path)
            logging.info(f"Test features shape: {X_test.shape}")
            print(f"   ✅ Test features shape: {X_test.shape}")
            
            # Fit and apply StandardScaler
            logging.info("Fitting StandardScaler on training features...")
            print("   Fitting StandardScaler...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save the preprocessor (scaler)
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=scaler
            )
            logging.info(f"Preprocessor saved to: {self.transformation_config.preprocessor_obj_file_path}")
            
            logging.info(">>> Data Transformation Completed <<<")
            print("   ✅ Data Transformation Completed!\n")
            
            return (
                X_train_scaled,
                y_train,
                X_test_scaled,
                y_test,
                self.transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
