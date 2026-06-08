import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Ensure necessary directories exist
os.makedirs(os.path.join(project_root, 'data', 'processed'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'reports'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'models', 'saved_models'), exist_ok=True)

from src.config import config
from src.data_ingestion.ingestor import DataIngestor, FeatureEngineer
from src.temporal_split.splitter import TemporalSplitter
from src.eda.analyzer import EDAAnalyzer
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.continual_learning.learner import ContinualLearner
from src.explainability.explainer import ModelExplainer

def run_pipeline():
    print("--- Starting Healthcare ML Pipeline ---")
    
    # 1. Data Ingestion & Feature Engineering
    print("Step 1: Data Ingestion & Feature Engineering...")
    ingestor = DataIngestor()
    demo, obs, cond = ingestor.load_data()
    fe = FeatureEngineer()
    obs_agg = fe.aggregate_observations(obs)
    merged_df = fe.merge_datasets(demo, obs_agg, cond)
    
    # 2. Temporal Split
    print("Step 2: Temporal Splitting...")
    splitter = TemporalSplitter()
    d1, d2 = splitter.split(merged_df)
    
    # Save processed datasets using relative paths
    d1_path = os.path.join(project_root, 'data', 'processed', 'dataset1_historical.csv')
    d2_path = os.path.join(project_root, 'data', 'processed', 'dataset2_current.csv')
    d1.to_csv(d1_path, index=False)
    d2.to_csv(d2_path, index=False)
    print(f"Saved Dataset 1 to {d1_path}")
    print(f"Saved Dataset 2 to {d2_path}")
    
    # 3. EDA & Drift Analysis
    print("Step 3: EDA & Drift Analysis...")
    analyzer = EDAAnalyzer()
    analyzer.plot_target_distribution(d1, d2)
    # Generate Box-plots for top numerical features
    analyzer.plot_feature_boxplots(d1, d2, config.NUMERICAL_FEATURES[:5])
    drift_df = analyzer.analyze_drift(d1, d2, config.NUMERICAL_FEATURES)
    drift_path = os.path.join(project_root, 'reports', 'drift_analysis.csv')
    drift_df.to_csv(drift_path, index=False)
    print(f"Saved drift analysis to {drift_path}")
    
    # 4. Train/Test Split (Dataset 1)
    print("Step 4: Training Models on Dataset 1...")
    X1 = d1[config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES]
    y1 = d1[config.TARGET_COLUMN]
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    
    trainer = ModelTrainer()
    trained_models = trainer.train_all(X1_train, y1_train)
    
    # 5. Evaluation (Dataset 1 & Dataset 2)
    print("Step 5: Evaluating Models...")
    evaluator = ModelEvaluator()
    results = []
    
    # Prepare Dataset 2 test set
    X2 = d2[config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES]
    y2 = d2[config.TARGET_COLUMN]
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    
    for name, model in trained_models.items():
        # Eval on D1
        m1 = evaluator.evaluate(model, X1_test, y1_test, name, 'D1_Test')
        results.append(m1)
        # Eval on D2
        m2 = evaluator.evaluate(model, X2_test, y2_test, name, 'D2_Test')
        results.append(m2)
        
    evaluator.plot_roc_comparison(trained_models, X1_test, y1_test, 'D1_Test')
    evaluator.plot_roc_comparison(trained_models, X2_test, y2_test, 'D2_Test')
    
    # 6. Continual Learning
    print("Step 6: Continual Learning (Fine-tuning MLP)...")
    learner = ContinualLearner()
    cl_model = learner.fine_tune_mlp(X2_train, y2_train)
    
    # Evaluate CL model
    m_cl = evaluator.evaluate(cl_model, X2_test, y2_test, 'MLP_FineTuned', 'D2_Test')
    results.append(m_cl)
    
    perf_path = os.path.join(project_root, 'reports', 'model_performance.csv')
    pd.DataFrame(results).to_csv(perf_path, index=False)
    print(f"Saved model performance to {perf_path}")
    
    # 7. Interpretability
    print("Step 7: Generating Interpretability Reports...")
    explainer = ModelExplainer()
    
    # Define models to explain
    models_to_explain = {
        'DecisionTree': trained_models['DecisionTree'],
        'SVM': trained_models['SVM'],
        'MLP': trained_models['MLP'],
        'MLP_FineTuned': cl_model
    }
    
    for name, model in models_to_explain.items():
        print(f"Generating explanations for {name}...")
        # Use appropriate test set for importance
        X_curr = X2_test if "FineTuned" in name else X1_test
        y_curr = y2_test if "FineTuned" in name else y1_test
        
        # 1. Feature Importance for all models
        explainer.plot_feature_importance(model, X_curr.head(200), y_curr.head(200), name)
        
        # 2. SHAP Heatmap (using a sample to save time)
        explainer.generate_shap_heatmap(model, X_curr.head(50), name)
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    run_pipeline()
