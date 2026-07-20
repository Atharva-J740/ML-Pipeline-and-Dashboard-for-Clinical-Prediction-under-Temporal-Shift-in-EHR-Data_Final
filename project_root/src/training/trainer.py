"""Model training for classification tasks."""
import pandas as pd
import joblib
import os
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

from src.config.config import MODELS_DIR, RANDOM_SEED
from src.preprocessing.pipeline import get_preprocessing_pipeline


class ModelTrainer:
    """Trains multiple classification models on preprocessed data."""
    
    def __init__(self, models_dir=MODELS_DIR):
        """Initialize trainer with model directory.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.preprocessor = get_preprocessing_pipeline()
        logger.info(f"ModelTrainer initialized. Models dir: {models_dir}")
        
    def train_all(self, X_train, y_train):
        """Train Decision Tree, SVM, and MLP models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained model pipelines
        """
        logger.info(f"Training models on {X_train.shape[0]} samples...")
        
        # 1. Decision Tree
        dt_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=RANDOM_SEED))
        ])
        
        # 2. SVM
        svm_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', SVC(probability=True, random_state=RANDOM_SEED, kernel='rbf'))
        ])
        
        # 3. MLP
        mlp_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=RANDOM_SEED,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ])
        
        models = {
            'DecisionTree': dt_pipeline,
            'SVM': svm_pipeline,
            'MLP': mlp_pipeline
        }
        
        trained_models = {}
        for name, pipeline in models.items():
            try:
                logger.info(f"Training {name}...")
                pipeline.fit(X_train, y_train)
                joblib.dump(pipeline, os.path.join(self.models_dir, f'{name}_model.joblib'))
                trained_models[name] = pipeline
                logger.info(f"{name} trained and saved")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                raise
            
        return trained_models
