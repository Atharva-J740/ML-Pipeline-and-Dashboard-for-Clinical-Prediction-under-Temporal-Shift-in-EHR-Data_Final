import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import config

class ModelExplainer:
    def __init__(self, output_dir=config.REPORTS_DIR):
        self.output_dir = output_dir
        
    def _get_feature_names(self, preprocessor):
        """Helper to extract feature names from the preprocessor pipeline."""
        cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(config.CATEGORICAL_FEATURES)
        return config.NUMERICAL_FEATURES + list(cat_features)

    def generate_shap_heatmap(self, model_pipeline, X_sample, model_name):
        """Generates a SHAP heatmap for the model predictions."""
        classifier = model_pipeline.named_steps['classifier']
        preprocessor = model_pipeline.named_steps['preprocessor']
        
        # Transform sample
        X_transformed = preprocessor.transform(X_sample)
        feature_names = self._get_feature_names(preprocessor)
        
        # Create SHAP explainer
        # For non-tree models, we use KernelExplainer or LinearExplainer
        try:
            if "DecisionTree" in model_name:
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_transformed)
            else:
                # For SVM and MLP, we use a subset for background to speed up
                background = shap.sample(X_transformed, 50) if X_transformed.shape[0] > 50 else X_transformed
                explainer = shap.KernelExplainer(classifier.predict_proba, background)
                shap_values = explainer.shap_values(X_transformed)
            
            # Extract values for the positive class (1)
            if isinstance(shap_values, list):
                # Binary classification often returns a list [neg_values, pos_values]
                values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                values = shap_values

            # Create SHAP Explanation object for the heatmap
            explanation = shap.Explanation(
                values=values,
                data=X_transformed,
                feature_names=feature_names
            )
            
            plt.figure(figsize=(12, 8))
            # Try a standard summary plot if heatmap fails with hclust error
            try:
                shap.plots.heatmap(explanation, show=False)
            except:
                shap.summary_plot(values, X_transformed, feature_names=feature_names, show=False)
            plt.title(f'SHAP Heatmap/Summary: {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'shap_heatmap_{model_name}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error generating SHAP heatmap for {model_name}: {e}")

    def plot_feature_importance(self, model_pipeline, X_test, y_test, model_name):
        """Calculates and plots feature importance for any model using permutation importance or coefficients."""
        classifier = model_pipeline.named_steps['classifier']
        preprocessor = model_pipeline.named_steps['preprocessor']
        feature_names = self._get_feature_names(preprocessor)
        
        X_transformed = preprocessor.transform(X_test)
        
        # 1. Try intrinsic importance (Tree-based)
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            method = "Intrinsic"
        # 2. Try coefficients (Linear models like SVM with linear kernel)
        elif hasattr(classifier, 'coef_'):
            importances = np.abs(classifier.coef_[0])
            method = "Coefficients"
        # 3. Fallback to Permutation Importance (MLP, Non-linear SVM)
        else:
            result = permutation_importance(classifier, X_transformed, y_test, n_repeats=5, random_state=config.RANDOM_SEED)
            importances = result.importances_mean
            method = "Permutation"
            
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        feat_imp.head(20).plot(kind='barh', color='skyblue')
        plt.title(f'Feature Importance ({method}): {model_name}')
        plt.xlabel('Importance Score')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'feature_importance_{model_name}.png'))
        plt.close()
        
        return feat_imp
