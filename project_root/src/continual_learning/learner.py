import joblib
import os
import sys
from sklearn.neural_network import MLPClassifier
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import config

class ContinualLearner:
    def __init__(self, models_dir=config.MODELS_DIR):
        self.models_dir = models_dir
        
    def fine_tune_mlp(self, X_train_new, y_train_new):
        # Load the base MLP model
        mlp_path = os.path.join(self.models_dir, 'MLP_model.joblib')
        if not os.path.exists(mlp_path):
            raise FileNotFoundError("Base MLP model not found for fine-tuning.")
            
        pipeline = joblib.load(mlp_path)
        mlp = pipeline.named_steps['classifier']
        
        # Fine-tune by setting warm_start=True and calling fit again
        mlp.warm_start = True
        
        # Preprocess the new data using the existing preprocessor
        X_processed = pipeline.named_steps['preprocessor'].transform(X_train_new)
        
        # Partial fit or fit with warm_start
        mlp.fit(X_processed, y_train_new)
        
        # Save the updated model
        updated_path = os.path.join(self.models_dir, '../continual_learning/MLP_fine_tuned.joblib')
        os.makedirs(os.path.dirname(updated_path), exist_ok=True)
        joblib.dump(pipeline, updated_path)
        
        return pipeline
