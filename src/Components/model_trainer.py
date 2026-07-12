import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    """Configuration for model training."""
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    metrics_file_path: str = os.path.join('artifacts', 'metrics.pkl')


class ModelTrainer:
    """
    Trains an SVM classifier with hyperparameter tuning via GridSearchCV.
    
    Tunes:
    - C (regularization): [0.1, 1, 10]
    - gamma (kernel coefficient): ['scale', 'auto', 0.01, 0.1]
    - kernel: ['rbf', 'linear']
    """
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Train SVM model with GridSearchCV and evaluate on test data.
        
        Args:
            X_train: Scaled training features
            y_train: Training labels
            X_test: Scaled test features
            y_test: Test labels
        
        Returns:
            Dictionary with model metrics and best parameters
        """
        try:
            logging.info(">>> Model Training Started <<<")
            print("\n" + "=" * 60)
            print("🤖 STEP 3: Model Training (SVM + GridSearchCV)")
            print("=" * 60)
            
            # Define SVM and hyperparameter grid
            svm = SVC(probability=True, random_state=42)
            
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01],
                'kernel': ['rbf', 'linear']
            }
            
            logging.info(f"Starting GridSearchCV with param_grid: {param_grid}")
            logging.info(f"Training data shape: {X_train.shape}")
            print(f"   Training data shape: {X_train.shape}")
            print(f"   Param grid: {param_grid}")
            print("   Running GridSearchCV (this may take a while)...")
            
            # GridSearchCV with 3-fold cross-validation
            grid_search = GridSearchCV(
                svm,
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                refit=True
            )
            
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
            
            logging.info(f"Best Parameters: {best_params}")
            logging.info(f"Best CV Score: {best_cv_score:.4f}")
            print(f"   ✅ Best Parameters: {best_params}")
            print(f"   ✅ Best CV Score: {best_cv_score:.4f}")
            
            # Evaluate on test data
            results = evaluate_model(best_model, X_test, y_test)
            results['best_params'] = best_params
            results['best_cv_score'] = best_cv_score
            
            logging.info(f"Test Accuracy: {results['accuracy']:.4f}")
            print(f"   ✅ Test Accuracy: {results['accuracy']:.4f}")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Model saved to: {self.model_trainer_config.trained_model_file_path}")
            
            # Save metrics
            save_object(
                file_path=self.model_trainer_config.metrics_file_path,
                obj=results
            )
            logging.info(f"Metrics saved to: {self.model_trainer_config.metrics_file_path}")
            
            logging.info(">>> Model Training Completed <<<")
            print("   ✅ Model Training Completed!\n")
            
            return results
        
        except Exception as e:
            raise CustomException(e, sys)
