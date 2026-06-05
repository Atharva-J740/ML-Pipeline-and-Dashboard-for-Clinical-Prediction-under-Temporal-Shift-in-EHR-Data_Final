import pandas as pd
import joblib
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import config
from src.preprocessing.pipeline import get_preprocessing_pipeline

class ModelTrainer:
    def __init__(self, models_dir=config.MODELS_DIR):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.preprocessor = get_preprocessing_pipeline()
        
    def train_all(self, X_train, y_train):
        # 1. Decision Tree
        dt_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=config.RANDOM_SEED))
        ])
        
        # 2. SVM
        svm_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', SVC(probability=True, random_state=config.RANDOM_SEED))
        ])
        
        # 3. MLP
        mlp_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', MLPClassifier(max_iter=500, random_state=config.RANDOM_SEED))
        ])
        
        models = {
            'DecisionTree': dt_pipeline,
            'SVM': svm_pipeline,
            'MLP': mlp_pipeline
        }
        
        trained_models = {}
        for name, pipeline in models.items():
            print(f"Training {name}...")
            pipeline.fit(X_train, y_train)
            joblib.dump(pipeline, os.path.join(self.models_dir, f'{name}_model.joblib'))
            trained_models[name] = pipeline
            
        return trained_models

from sklearn.pipeline import Pipeline
