import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import config

class EDAAnalyzer:
    def __init__(self, output_dir=config.REPORTS_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_target_distribution(self, d1, d2):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.countplot(x='condition_binary', data=d1, palette='viridis')
        plt.title('Dataset 1 (Historical) Target Distribution')
        
        plt.subplot(1, 2, 2)
        sns.countplot(x='condition_binary', data=d2, palette='magma')
        plt.title('Dataset 2 (Current) Target Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'target_distribution.png'))
        plt.close()

    def plot_feature_boxplots(self, d1, d2, features):
        """Generates box-plots to compare feature distributions across datasets and targets."""
        # Combine datasets for easier plotting
        d1_plot = d1.copy()
        d1_plot['Dataset'] = 'Historical (D1)'
        d2_plot = d2.copy()
        d2_plot['Dataset'] = 'Current (D2)'
        combined = pd.concat([d1_plot, d2_plot])
        
        for feat in features:
            if feat in combined.columns:
                plt.figure(figsize=(12, 6))
                
                # Subplot 1: By Dataset
                plt.subplot(1, 2, 1)
                sns.boxplot(x='Dataset', y=feat, data=combined, palette='Set2')
                plt.title(f'{feat} by Dataset')
                
                # Subplot 2: By Target within combined data
                plt.subplot(1, 2, 2)
                sns.boxplot(x='condition_binary', y=feat, hue='Dataset', data=combined, palette='Set1')
                plt.title(f'{feat} by Target & Dataset')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'boxplot_{feat}.png'))
                plt.close()

    def analyze_drift(self, d1, d2, features):
        drift_results = []
        for feat in features:
            if feat in d1.columns and feat in d2.columns:
                # KS test for numerical features
                stat, p_val = ks_2samp(d1[feat].dropna(), d2[feat].dropna())
                drift_results.append({
                    'feature': feat,
                    'ks_stat': stat,
                    'p_value': p_val,
                    'drift_detected': p_val < 0.05
                })
        return pd.DataFrame(drift_results)

    def plot_feature_drift(self, d1, d2, feature):
        plt.figure(figsize=(8, 5))
        sns.kdeplot(d1[feature], label='Historical (D1)', fill=True)
        sns.kdeplot(d2[feature], label='Current (D2)', fill=True)
        plt.title(f'Distribution Drift: {feature}')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'drift_{feature}.png'))
        plt.close()
