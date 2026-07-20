"""Model evaluation and performance metrics."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import os
import logging

logger = logging.getLogger(__name__)

from src.config.config import REPORTS_DIR


class ModelEvaluator:
    """Evaluates model performance on test datasets."""
    
    def __init__(self, output_dir=REPORTS_DIR):
        """Initialize evaluator with output directory.
        
        Args:
            output_dir: Directory to save evaluation plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"ModelEvaluator initialized. Output dir: {output_dir}")
        
    def evaluate(self, model, X_test, y_test, model_name, dataset_name):
        """Evaluate model on test data and compute metrics.
        
        Args:
            model: Trained model pipeline
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            dataset_name: Name of the dataset (e.g., 'D1_Test', 'D2_Test')
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            logger.info(f"Evaluating {model_name} on {dataset_name}...")
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'model': model_name,
                'dataset': dataset_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            logger.info(f"Metrics: {metrics}")
            
            # Save confusion matrix
            try:
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.output_dir, f'cm_{model_name}_{dataset_name}.png'),
                    dpi=100,
                    bbox_inches='tight'
                )
                plt.close()
            except Exception as e:
                logger.warning(f"Could not save confusion matrix: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            raise

    def plot_roc_comparison(self, model_results, X_test, y_test, dataset_name):
        """Plot ROC curves for multiple models.
        
        Args:
            model_results: Dictionary of model pipelines
            X_test: Test features
            y_test: Test labels
            dataset_name: Name of the dataset
        """
        try:
            logger.info(f"Plotting ROC curves for {dataset_name}...")
            plt.figure(figsize=(10, 8))
            
            for name, model in model_results.items():
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
                except Exception as e:
                    logger.warning(f"Could not plot ROC for {name}: {e}")
            
            # Diagonal line
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Comparison: {dataset_name}', fontsize=14)
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(
                os.path.join(self.output_dir, f'roc_comparison_{dataset_name}.png'),
                dpi=100,
                bbox_inches='tight'
            )
            plt.close()
            logger.info(f"ROC plot saved for {dataset_name}")
        except Exception as e:
            logger.error(f"Error plotting ROC curves: {e}")
