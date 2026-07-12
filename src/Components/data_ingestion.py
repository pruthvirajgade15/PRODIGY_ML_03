import os
import sys
import shutil
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths."""
    raw_data_path: str = os.path.join('artifacts', 'raw_data')
    train_data_path: str = os.path.join('artifacts', 'raw_data', 'train')
    test_data_path: str = os.path.join('artifacts', 'raw_data', 'test')


class DataIngestion:
    """
    Handles organizing the Cats vs Dogs dataset from local extracted images.
    
    Scans the raw_data directory recursively for cat/dog images and
    organizes them into train/test splits.
    """
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def extract_and_organize(self) -> tuple:
        """
        Scan raw_data directory for cat/dog images and organize into
        train/test directories with an 80/20 split.
        
        Returns:
            Tuple of (train_path, test_path)
        """
        try:
            logging.info("Scanning and organizing dataset...")
            print("🔄 Scanning and organizing dataset...")
            
            extract_path = self.ingestion_config.raw_data_path
            
            # Find all image paths recursively inside extract_path
            cat_images = []
            dog_images = []
            
            # Use os.sep-terminated paths to avoid false prefix matches
            # (e.g., "training_set" falsely matching output dir "train")
            norm_train_path = os.path.abspath(self.ingestion_config.train_data_path) + os.sep
            norm_test_path = os.path.abspath(self.ingestion_config.test_data_path) + os.sep
            
            for root, dirs, files in os.walk(extract_path):
                abs_root = os.path.abspath(root) + os.sep
                # Skip the organized output directories to avoid double-counting
                if abs_root.startswith(norm_train_path) or abs_root.startswith(norm_test_path):
                    continue
                
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_img_path = os.path.join(root, f)
                        parent_dir = os.path.basename(root).lower()
                        filename = f.lower()
                        
                        if filename.startswith('cat') or 'cat' in parent_dir:
                            cat_images.append(full_img_path)
                        elif filename.startswith('dog') or 'dog' in parent_dir:
                            dog_images.append(full_img_path)
            
            if not cat_images and not dog_images:
                raise ValueError(
                    f"No cat or dog images found in {extract_path}.\n"
                    "Please ensure the dataset is extracted in the artifacts/raw_data directory.\n"
                    "Expected image filenames starting with 'cat' or 'dog', "
                    "or images inside folders named 'cats' or 'dogs'."
                )
            
            logging.info(f"Found {len(cat_images)} cat images and {len(dog_images)} dog images total")
            print(f"   Found {len(cat_images)} cat images and {len(dog_images)} dog images")
            
            # Split into train and test (80/20)
            cat_train, cat_test = train_test_split(cat_images, test_size=0.2, random_state=42)
            dog_train, dog_test = train_test_split(dog_images, test_size=0.2, random_state=42)
            
            # Clean up any stale organized directories before copying
            train_path = self.ingestion_config.train_data_path
            test_path = self.ingestion_config.test_data_path
            
            for path in [train_path, test_path]:
                if os.path.exists(path):
                    shutil.rmtree(path)
            
            # Create organized directories
            for path in [
                os.path.join(train_path, 'cats'),
                os.path.join(train_path, 'dogs'),
                os.path.join(test_path, 'cats'),
                os.path.join(test_path, 'dogs'),
            ]:
                os.makedirs(path, exist_ok=True)
            
            # Copy images to organized structure
            print("   Copying images to organized directories...")
            self._copy_images(cat_train, os.path.join(train_path, 'cats'))
            self._copy_images(cat_test, os.path.join(test_path, 'cats'))
            self._copy_images(dog_train, os.path.join(train_path, 'dogs'))
            self._copy_images(dog_test, os.path.join(test_path, 'dogs'))
            
            logging.info(f"Train path: {train_path} (cats: {len(cat_train)}, dogs: {len(dog_train)})")
            logging.info(f"Test path: {test_path} (cats: {len(cat_test)}, dogs: {len(dog_test)})")
            print(f"   ✅ Train: {len(cat_train)} cats + {len(dog_train)} dogs")
            print(f"   ✅ Test:  {len(cat_test)} cats + {len(dog_test)} dogs")
            
            return train_path, test_path
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def _copy_images(self, image_list: list, dest_dir: str) -> None:
        """Copy a list of images to destination directory."""
        for src in image_list:
            if not os.path.exists(src):
                continue
            img_name = os.path.basename(src)
            dst = os.path.join(dest_dir, img_name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    
    def initiate_data_ingestion(self) -> tuple:
        """
        Complete data ingestion pipeline.
        
        Checks if data is already organized; if not, scans raw_data
        directory for images and organizes them into train/test splits.
        
        Returns:
            Tuple of (train_path, test_path)
        """
        try:
            logging.info(">>> Data Ingestion Started <<<")
            print("\n" + "=" * 60)
            print("📦 STEP 1: Data Ingestion")
            print("=" * 60)
            
            # Check if data already exists and is organized
            train_cats = os.path.join(self.ingestion_config.train_data_path, 'cats')
            train_dogs = os.path.join(self.ingestion_config.train_data_path, 'dogs')
            if (os.path.exists(train_cats) and len(os.listdir(train_cats)) > 0 and
                    os.path.exists(train_dogs) and len(os.listdir(train_dogs)) > 0):
                cat_count = len(os.listdir(train_cats))
                dog_count = len(os.listdir(train_dogs))
                logging.info("Dataset already organized. Skipping.")
                print(f"   ✅ Dataset already organized ({cat_count} cats, {dog_count} dogs). Skipping.")
                return (
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                )
            
            # Organize from local extracted data
            train_path, test_path = self.extract_and_organize()
            
            logging.info(">>> Data Ingestion Completed <<<")
            print("   ✅ Data Ingestion Completed!\n")
            return train_path, test_path
        
        except Exception as e:
            raise CustomException(e, sys)
