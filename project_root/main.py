#!/usr/bin/env python3
"""Healthcare ML Pipeline - Complete Orchestration.

This script runs the full ML pipeline:
1. Data ingestion and feature engineering
2. Temporal splitting
3. EDA and drift analysis
4. Model training
5. Cross-temporal evaluation
6. Continual learning
7. Model explainability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import pipeline components
from src.config.config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
    TEST_SIZE, RANDOM_SEED, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR
)
from src.data_ingestion.ingestor import DataIngestor, FeatureEngineer
from src.temporal_split.splitter import TemporalSplitter
from src.eda.analyzer import EDAAnalyzer
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.continual_learning.learner import ContinualLearner
from src.explainability.explainer import ModelExplainer


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ready: {directory}")


def run_pipeline():
    """Execute the complete ML pipeline."""
    try:
        logger.info("="*80)
        logger.info("Starting Healthcare ML Pipeline")
        logger.info("="*80)
        
        # Ensure directories
        ensure_directories()
        
        # Step 1: Data Ingestion & Feature Engineering
        logger.info("\n[STEP 1] Data Ingestion & Feature Engineering...")
        ingestor = DataIngestor()
        patients, obs, cond = ingestor.load_data()
        logger.info(f"  - Patients: {patients.shape}")
        logger.info(f"  - Observations: {obs.shape}")
        logger.info(f"  - Conditions: {cond.shape}")
        
        fe = FeatureEngineer()
        obs_agg = fe.aggregate_observations(obs)
        merged_df = fe.merge_datasets(patients, obs_agg, cond)
        logger.info(f"  - Merged dataset: {merged_df.shape}")
        
        # Step 2: Temporal Split
        logger.info("\n[STEP 2] Temporal Splitting...")
        splitter = TemporalSplitter()
        d1, d2 = splitter.split(merged_df)
        logger.info(f"  - Dataset 1 (Historical): {d1.shape}")
        logger.info(f"  - Dataset 2 (Current): {d2.shape}")
        
        # Save processed datasets
        d1_path = os.path.join(PROCESSED_DATA_DIR, 'dataset1_historical.csv')
        d2_path = os.path.join(PROCESSED_DATA_DIR, 'dataset2_current.csv')
        d1.to_csv(d1_path, index=False)
        d2.to_csv(d2_path, index=False)
        logger.info(f"  - Saved D1 to {d1_path}")
        logger.info(f"  - Saved D2 to {d2_path}")
        
        # Step 3: EDA & Drift Analysis
        logger.info("\n[STEP 3] EDA & Drift Analysis...")
        analyzer = EDAAnalyzer()
        analyzer.plot_target_distribution(d1, d2)
        
        # Plot box-plots for top features
        valid_numerical = [f for f in NUMERICAL_FEATURES if f in d1.columns]
        if valid_numerical:
            analyzer.plot_feature_boxplots(d1, d2, valid_numerical[:5])
        
        drift_df = analyzer.analyze_drift(d1, d2, valid_numerical)
        drift_path = os.path.join(REPORTS_DIR, 'drift_analysis.csv')
        drift_df.to_csv(drift_path, index=False)
        logger.info(f"  - Drift analysis saved to {drift_path}")
        logger.info(f"  - Features with drift: {drift_df['drift_detected'].sum()}")
        
        # Step 4: Train Models on Dataset 1
        logger.info("\n[STEP 4] Training Models on Dataset 1...")
        
        # Prepare training data
        valid_features = [f for f in (NUMERICAL_FEATURES + CATEGORICAL_FEATURES) if f in d1.columns]
        X1 = d1[valid_features].copy()
        y1 = d1[TARGET_COLUMN].copy()
        
        X1_train, X1_test, y1_train, y1_test = train_test_split(
            X1, y1,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED
        )
        logger.info(f"  - D1 Train: {X1_train.shape}, Test: {X1_test.shape}")
        
        trainer = ModelTrainer()
        trained_models = trainer.train_all(X1_train, y1_train)
        logger.info(f"  - Trained {len(trained_models)} models")
        
        # Step 5: Cross-Temporal Evaluation
        logger.info("\n[STEP 5] Cross-Temporal Evaluation...")
        evaluator = ModelEvaluator()
        results = []
        
        # Evaluate on D1 test set
        for name, model in trained_models.items():
            m1 = evaluator.evaluate(model, X1_test, y1_test, name, 'D1_Test')
            results.append(m1)
            logger.info(f"  - {name} on D1: Accuracy={m1['accuracy']:.3f}, F1={m1['f1']:.3f}")
        
        # Evaluate on D2 test set
        X2 = d2[valid_features].copy()
        y2 = d2[TARGET_COLUMN].copy()
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED
        )
        logger.info(f"  - D2 Train: {X2_train.shape}, Test: {X2_test.shape}")
        
        for name, model in trained_models.items():
            m2 = evaluator.evaluate(model, X2_test, y2_test, name, 'D2_Test')
            results.append(m2)
            logger.info(f"  - {name} on D2: Accuracy={m2['accuracy']:.3f}, F1={m2['f1']:.3f}")
        
        # Plot ROC curves
        try:
            evaluator.plot_roc_comparison(trained_models, X1_test, y1_test, 'D1_Test')
            evaluator.plot_roc_comparison(trained_models, X2_test, y2_test, 'D2_Test')
            logger.info("  - ROC plots generated")
        except Exception as e:
            logger.warning(f"  - Could not generate ROC plots: {e}")
        
        # Step 6: Continual Learning
        logger.info("\n[STEP 6] Continual Learning (Fine-tuning MLP)...")
        try:
            learner = ContinualLearner()
            cl_model = learner.fine_tune_mlp(X2_train, y2_train)
            m_cl = evaluator.evaluate(cl_model, X2_test, y2_test, 'MLP_FineTuned', 'D2_Test')
            results.append(m_cl)
            logger.info(f"  - Fine-tuned MLP: Accuracy={m_cl['accuracy']:.3f}, F1={m_cl['f1']:.3f}")
        except Exception as e:
            logger.warning(f"  - Continual learning skipped: {e}")
        
        # Save model performance
        perf_df = pd.DataFrame(results)
        perf_path = os.path.join(REPORTS_DIR, 'model_performance.csv')
        perf_df.to_csv(perf_path, index=False)
        logger.info(f"  - Performance metrics saved to {perf_path}")
        
        # Step 7: Model Explainability
        logger.info("\n[STEP 7] Model Explainability...")
        explainer = ModelExplainer()
        
        models_to_explain = {
            'DecisionTree': trained_models.get('DecisionTree'),
            'SVM': trained_models.get('SVM'),
            'MLP': trained_models.get('MLP'),
        }
        
        # Add fine-tuned MLP if available
        try:
            models_to_explain['MLP_FineTuned'] = cl_model
        except:
            pass
        
        for name, model in models_to_explain.items():
            if model is None:
                continue
            try:
                logger.info(f"  - Generating explanations for {name}...")
                
                # Choose appropriate test set
                X_curr = X2_test if "FineTuned" in name else X1_test
                y_curr = y2_test if "FineTuned" in name else y1_test
                
                # Feature importance
                explainer.plot_feature_importance(
                    model,
                    X_curr.head(min(200, len(X_curr))),
                    y_curr.head(min(200, len(y_curr))),
                    name
                )
                
                # SHAP heatmap
                explainer.generate_shap_heatmap(
                    model,
                    X_curr.head(min(50, len(X_curr))),
                    name
                )
                logger.info(f"    - {name} explanations complete")
            except Exception as e:
                logger.warning(f"    - Could not generate explanations for {name}: {e}")
        
        logger.info("\n" + "="*80)
        logger.info("Pipeline Completed Successfully!")
        logger.info("="*80)
        logger.info(f"Reports saved to: {REPORTS_DIR}")
        logger.info(f"Models saved to: {MODELS_DIR}")
        logger.info(f"\nNext: Run 'streamlit run dashboard/dashboard.py'")
        
        return True
        
    except Exception as e:
        logger.error(f"\n" + "="*80)
        logger.error(f"Pipeline Failed with Error: {e}")
        logger.error("="*80)
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
