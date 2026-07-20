"""Exploratory Data Analysis and drift detection."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import os
import logging

logger = logging.getLogger(__name__)

from src.config.config import REPORTS_DIR


class EDAAnalyzer:
    """Performs exploratory data analysis and temporal drift detection."""
    
    def __init__(self, output_dir=REPORTS_DIR):
        """Initialize analyzer with output directory.
        
        Args:
            output_dir: Directory to save analysis plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"EDAAnalyzer initialized. Output dir: {output_dir}")
        
    def plot_target_distribution(self, d1, d2):
        """Plot target variable distribution across datasets.
        
        Args:
            d1: Historical dataset
            d2: Current dataset
        """
        logger.info("Plotting target distribution...")
        try:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            d1['condition_binary'].value_counts().plot(kind='bar', color=['green', 'red'])
            plt.title('Dataset 1 (Historical) Target Distribution')
            plt.xlabel('Condition (0=Healthy, 1=Diabetes)')
            plt.ylabel('Count')
            
            plt.subplot(1, 2, 2)
            d2['condition_binary'].value_counts().plot(kind='bar', color=['green', 'red'])
            plt.title('Dataset 2 (Current) Target Distribution')
            plt.xlabel('Condition (0=Healthy, 1=Diabetes)')
            plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'target_distribution.png'), dpi=100, bbox_inches='tight')
            plt.close()
            logger.info("Target distribution plot saved")
        except Exception as e:
            logger.error(f"Error plotting target distribution: {e}")

    def plot_feature_boxplots(self, d1, d2, features):
        """Generate box-plots comparing feature distributions across datasets.
        
        Args:
            d1: Historical dataset
            d2: Current dataset
            features: List of feature names to plot
        """
        logger.info(f"Plotting box-plots for {len(features)} features...")
        
        # Combine datasets for easier plotting
        d1_plot = d1.copy()
        d1_plot['Dataset'] = 'Historical (D1)'
        d2_plot = d2.copy()
        d2_plot['Dataset'] = 'Current (D2)'
        combined = pd.concat([d1_plot, d2_plot], ignore_index=True)
        
        for feat in features:
            if feat not in combined.columns:
                logger.warning(f"Feature {feat} not found in data")
                continue
            
            try:
                plt.figure(figsize=(14, 6))
                
                # Subplot 1: By Dataset
                plt.subplot(1, 2, 1)
                combined.boxplot(column=feat, by='Dataset', ax=plt.gca())
                plt.title(f'{feat} Distribution by Dataset')
                plt.suptitle('')
                
                # Subplot 2: By Target & Dataset
                plt.subplot(1, 2, 2)
                combined_valid = combined.dropna(subset=[feat, 'condition_binary'])
                if len(combined_valid) > 0:
                    combined_valid.boxplot(
                        column=feat,
                        by=['condition_binary', 'Dataset'],
                        ax=plt.gca()
                    )
                    plt.title(f'{feat} by Target & Dataset')
                    plt.suptitle('')
                
                plt.tight_layout()
                safe_name = feat.replace('/', '_').replace(' ', '_')
                plt.savefig(
                    os.path.join(self.output_dir, f'boxplot_{safe_name}.png'),
                    dpi=100,
                    bbox_inches='tight'
                )
                plt.close()
                logger.info(f"Box-plot saved for {feat}")
            except Exception as e:
                logger.error(f"Error plotting box-plot for {feat}: {e}")

    def analyze_drift(self, d1, d2, features):
        """Detect temporal distribution shift using Kolmogorov-Smirnov test.
        
        Args:
            d1: Historical dataset
            d2: Current dataset
            features: List of numerical features to analyze
            
        Returns:
            DataFrame with drift statistics for each feature
        """
        logger.info(f"Analyzing drift for {len(features)} features...")
        drift_results = []
        
        for feat in features:
            if feat not in d1.columns or feat not in d2.columns:
                logger.warning(f"Feature {feat} not found in both datasets")
                continue
            
            try:
                d1_valid = d1[feat].dropna()
                d2_valid = d2[feat].dropna()
                
                if len(d1_valid) > 0 and len(d2_valid) > 0:
                    stat, p_val = ks_2samp(d1_valid, d2_valid)
                    drift_results.append({
                        'feature': feat,
                        'ks_stat': stat,
                        'p_value': p_val,
                        'drift_detected': p_val < 0.05
                    })
            except Exception as e:
                logger.error(f"Error analyzing drift for {feat}: {e}")
        
        drift_df = pd.DataFrame(drift_results)
        logger.info(f"Drift detected in {drift_df['drift_detected'].sum()} features")
        return drift_df

    def plot_feature_drift(self, d1, d2, feature):
        """Visualize distribution drift for a specific feature.
        
        Args:
            d1: Historical dataset
            d2: Current dataset
            feature: Feature name to visualize
        """
        try:
            plt.figure(figsize=(10, 6))
            
            d1[feature].dropna().hist(bins=30, alpha=0.6, label='Historical (D1)', density=True)
            d2[feature].dropna().hist(bins=30, alpha=0.6, label='Current (D2)', density=True)
            
            plt.title(f'Distribution Drift: {feature}')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            
            safe_name = feature.replace('/', '_').replace(' ', '_')
            plt.savefig(
                os.path.join(self.output_dir, f'drift_{safe_name}.png'),
                dpi=100,
                bbox_inches='tight'
            )
            plt.close()
            logger.info(f"Drift plot saved for {feature}")
        except Exception as e:
            logger.error(f"Error plotting drift for {feature}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_ingestion.ingestor import DataIngestor, FeatureEngineer
    from src.temporal_split.splitter import TemporalSplitter
    from src.config.config import NUMERICAL_FEATURES
    
    ingestor = DataIngestor()
    patients, obs, cond = ingestor.load_data()
    
    fe = FeatureEngineer()
    obs_agg = fe.aggregate_observations(obs)
    merged_df = fe.merge_datasets(patients, obs_agg, cond)
    
    splitter = TemporalSplitter()
    d1, d2 = splitter.split(merged_df)
    
    analyzer = EDAAnalyzer()
    analyzer.plot_target_distribution(d1, d2)
    analyzer.plot_feature_boxplots(d1, d2, NUMERICAL_FEATURES[:5])
    drift_df = analyzer.analyze_drift(d1, d2, NUMERICAL_FEATURES)
    print(f"\nDrift Analysis:\n{drift_df}")
