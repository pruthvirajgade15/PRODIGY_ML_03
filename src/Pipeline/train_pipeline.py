import sys

from src.exception import CustomException
from src.logger import logging
from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer


class TrainPipeline:
    """
    End-to-end training pipeline.
    
    Orchestrates: DataIngestion → DataTransformation → ModelTrainer
    """
    
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    
    def run_pipeline(self, max_samples: int = 2000) -> dict:
        """
        Execute the complete training pipeline.
        
        Args:
            max_samples: Maximum samples per class for training
        
        Returns:
            Dictionary with training results and metrics
        """
        try:
            logging.info("=" * 60)
            logging.info("TRAINING PIPELINE STARTED")
            logging.info("=" * 60)
            
            print("\n" + "🐾" * 30)
            print("  PRODIGY_ML_03 — SVM Cats vs Dogs Training Pipeline")
            print("🐾" * 30)
            
            # Update max samples in transformation config
            self.data_transformation.transformation_config.max_samples_per_class = max_samples
            
            # Step 1: Data Ingestion
            logging.info("STEP 1: Data Ingestion")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            
            # Step 2: Data Transformation
            logging.info("STEP 2: Data Transformation")
            X_train, y_train, X_test, y_test, preprocessor_path = (
                self.data_transformation.initiate_data_transformation(train_path, test_path)
            )
            
            # Step 3: Model Training
            logging.info("STEP 3: Model Training")
            results = self.model_trainer.initiate_model_trainer(
                X_train, y_train, X_test, y_test
            )
            
            logging.info("=" * 60)
            logging.info(f"TRAINING PIPELINE COMPLETED — Accuracy: {results['accuracy']:.4f}")
            logging.info("=" * 60)
            
            print("\n" + "=" * 60)
            print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"   Accuracy:        {results['accuracy'] * 100:.2f}%")
            print(f"   Best CV Score:   {results['best_cv_score'] * 100:.2f}%")
            print(f"   Best Parameters: {results['best_params']}")
            print("=" * 60 + "\n")
            
            return results
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the Cat vs Dog SVM classifier pipeline.")
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum samples per class to train on.")
    args = parser.parse_args()

    pipeline = TrainPipeline()
    results = pipeline.run_pipeline(max_samples=args.max_samples)
    print(f"\nTraining Complete!")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Best Parameters: {results['best_params']}")
    print(f"\nClassification Report:\n{results['classification_report_text']}")
