"""Continual learning and model fine-tuning."""
import joblib
import os
import logging

logger = logging.getLogger(__name__)

from src.config.config import MODELS_DIR


class ContinualLearner:
    """Fine-tunes models on new data to adapt to distribution shift."""
    
    def __init__(self, models_dir=MODELS_DIR):
        """Initialize continual learner.
        
        Args:
            models_dir: Directory containing trained base models
        """
        self.models_dir = models_dir
        logger.info(f"ContinualLearner initialized. Models dir: {models_dir}")
        
    def fine_tune_mlp(self, X_train_new, y_train_new):
        """Fine-tune MLP model on new data using warm_start.
        
        Args:
            X_train_new: New training features
            y_train_new: New training labels
            
        Returns:
            Updated model pipeline
            
        Raises:
            FileNotFoundError: If base MLP model not found
        """
        try:
            logger.info("Fine-tuning MLP on new data...")
            
            mlp_path = os.path.join(self.models_dir, 'MLP_model.joblib')
            if not os.path.exists(mlp_path):
                raise FileNotFoundError(f"Base MLP model not found at {mlp_path}")
            
            # Load base model
            pipeline = joblib.load(mlp_path)
            mlp_classifier = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocessor']
            
            # Enable warm_start for incremental learning
            mlp_classifier.warm_start = True
            mlp_classifier.n_iter_no_change_ = None  # Reset for new data
            
            # Preprocess new data
            X_processed = preprocessor.transform(X_train_new)
            
            logger.info(f"Fitting MLP with {X_processed.shape[0]} new samples...")
            mlp_classifier.fit(X_processed, y_train_new)
            
            # Save fine-tuned model
            cl_dir = os.path.join(self.models_dir, '..', 'continual_learning')
            os.makedirs(cl_dir, exist_ok=True)
            updated_path = os.path.join(cl_dir, 'MLP_fine_tuned.joblib')
            joblib.dump(pipeline, updated_path)
            logger.info(f"Fine-tuned MLP saved to {updated_path}")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Error fine-tuning MLP: {e}")
            raise
