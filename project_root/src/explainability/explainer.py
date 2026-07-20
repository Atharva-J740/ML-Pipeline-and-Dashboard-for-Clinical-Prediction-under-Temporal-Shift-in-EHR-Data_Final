"""Model explainability using SHAP and permutation importance."""
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

from src.config.config import REPORTS_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, RANDOM_SEED


class ModelExplainer:
    """Generates SHAP and feature importance explanations for models."""
    
    def __init__(self, output_dir=REPORTS_DIR):
        """Initialize explainer.
        
        Args:
            output_dir: Directory to save explanation plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"ModelExplainer initialized. Output dir: {output_dir}")
        
    def _get_feature_names(self, preprocessor):
        """Extract feature names from preprocessing pipeline.
        
        Args:
            preprocessor: sklearn ColumnTransformer
            
        Returns:
            List of feature names after preprocessing
        """
        try:
            cat_features = preprocessor.named_transformers_['cat'].named_steps[
                'onehot'
            ].get_feature_names_out(CATEGORICAL_FEATURES)
            return list(NUMERICAL_FEATURES) + list(cat_features)
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
            return list(NUMERICAL_FEATURES) + CATEGORICAL_FEATURES

    def generate_shap_heatmap(self, model_pipeline, X_sample, model_name):
        """Generate SHAP explanation heatmap.
        
        Args:
            model_pipeline: Trained model pipeline
            X_sample: Sample data for explanation
            model_name: Name of the model
        """
        try:
            logger.info(f"Generating SHAP heatmap for {model_name}...")
            
            classifier = model_pipeline.named_steps['classifier']
            preprocessor = model_pipeline.named_steps['preprocessor']
            
            # Transform sample
            X_transformed = preprocessor.transform(X_sample)
            feature_names = self._get_feature_names(preprocessor)
            
            # Create SHAP explainer
            try:
                if "DecisionTree" in model_name:
                    explainer = shap.TreeExplainer(classifier)
                    shap_values = explainer.shap_values(X_transformed)
                else:
                    # Use KernelExplainer for non-tree models
                    background_size = min(50, X_transformed.shape[0])
                    background = shap.sample(X_transformed, background_size)
                    explainer = shap.KernelExplainer(
                        classifier.predict_proba,
                        background
                    )
                    shap_values = explainer.shap_values(X_transformed)
                
                # Handle binary classification output format
                if isinstance(shap_values, list):
                    values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    values = shap_values
                
                # Create and save SHAP plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    values,
                    X_transformed,
                    feature_names=feature_names,
                    show=False
                )
                plt.title(f'SHAP Summary: {model_name}')
                plt.tight_layout()
                
                plt.savefig(
                    os.path.join(self.output_dir, f'shap_heatmap_{model_name}.png'),
                    dpi=100,
                    bbox_inches='tight'
                )
                plt.close()
                logger.info(f"SHAP heatmap saved for {model_name}")
                
            except Exception as e:
                logger.warning(f"Could not generate SHAP heatmap: {e}. Using bar plot instead.")
                
                # Fallback: simple bar plot
                plt.figure(figsize=(10, 8))
                values_mean = np.abs(values).mean(axis=0)
                top_features = np.argsort(values_mean)[-10:]
                
                plt.barh(range(len(top_features)), values_mean[top_features])
                plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
                plt.xlabel('Mean |SHAP value|')
                plt.title(f'SHAP Feature Importance: {model_name}')
                plt.tight_layout()
                
                plt.savefig(
                    os.path.join(self.output_dir, f'shap_heatmap_{model_name}.png'),
                    dpi=100,
                    bbox_inches='tight'
                )
                plt.close()
                
        except Exception as e:
            logger.error(f"Error generating SHAP heatmap for {model_name}: {e}")

    def plot_feature_importance(self, model_pipeline, X_test, y_test, model_name):
        """Plot feature importance for a model.
        
        Args:
            model_pipeline: Trained model pipeline
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Series of feature importances
        """
        try:
            logger.info(f"Computing feature importance for {model_name}...")
            
            classifier = model_pipeline.named_steps['classifier']
            preprocessor = model_pipeline.named_steps['preprocessor']
            feature_names = self._get_feature_names(preprocessor)
            
            X_transformed = preprocessor.transform(X_test)
            
            # Determine importance method
            if hasattr(classifier, 'feature_importances_'):
                # Tree-based models
                importances = classifier.feature_importances_
                method = "Intrinsic"
            elif hasattr(classifier, 'coef_'):
                # Linear models
                importances = np.abs(classifier.coef_[0])
                method = "Coefficients"
            else:
                # Fallback: Permutation importance
                result = permutation_importance(
                    classifier,
                    X_transformed,
                    y_test,
                    n_repeats=5,
                    random_state=RANDOM_SEED
                )
                importances = result.importances_mean
                method = "Permutation"
            
            # Sort and plot
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            
            plt.figure(figsize=(10, 8))
            feat_imp.head(20).plot(kind='barh', color='skyblue')
            plt.title(f'Feature Importance ({method}): {model_name}')
            plt.xlabel('Importance Score')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plt.savefig(
                os.path.join(self.output_dir, f'feature_importance_{model_name}.png'),
                dpi=100,
                bbox_inches='tight'
            )
            plt.close()
            logger.info(f"Feature importance plot saved for {model_name}")
            
            return feat_imp
            
        except Exception as e:
            logger.error(f"Error plotting feature importance for {model_name}: {e}")
            return None
