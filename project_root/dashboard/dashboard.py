#!/usr/bin/env python3
"""Healthcare ML Platform - Interactive Streamlit Dashboard.

This dashboard provides interactive visualization of:
- Data exploration (D1 vs D2)
- EDA and drift analysis
- Model training and evaluation
- Cross-temporal generalization
- Continual learning performance
- Model explainability (SHAP, feature importance)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.config import DATA_DIR, REPORTS_DIR, PROCESSED_DATA_DIR

# Page configuration
st.set_page_config(
    page_title="Healthcare ML Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-text {
        color: #1f77b4;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed datasets and analysis results with caching."""
    try:
        d1_path = os.path.join(PROCESSED_DATA_DIR, 'dataset1_historical.csv')
        d2_path = os.path.join(PROCESSED_DATA_DIR, 'dataset2_current.csv')
        perf_path = os.path.join(REPORTS_DIR, 'model_performance.csv')
        drift_path = os.path.join(REPORTS_DIR, 'drift_analysis.csv')
        
        # Check if files exist
        if not all(Path(p).exists() for p in [d1_path, d2_path, perf_path, drift_path]):
            return None, None, None, None, None
        
        d1 = pd.read_csv(d1_path)
        d2 = pd.read_csv(d2_path)
        perf = pd.read_csv(perf_path)
        drift = pd.read_csv(drift_path)
        
        return d1, d2, perf, drift, True
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None, None, False


def show_no_data_message():
    """Show message when data is not available."""
    st.warning("""
    📊 **Pipeline Data Not Available**
    
    It looks like the ML pipeline hasn't been run yet, or the data files are missing.
    
    To generate the required data and models:
    
    1. **Open a terminal** in the `project_root` directory
    2. **Run the pipeline**:
       ```bash
       python main.py
       ```
    3. **Wait for completion** (this may take a few minutes)
    4. **Refresh** this dashboard once complete
    
    The pipeline will:
    - Download data from Google Drive (first run only)
    - Process and prepare the EHR data
    - Train multiple classification models
    - Generate analysis plots and metrics
    - Create model explanations
    """)


# Sidebar navigation
st.sidebar.title("🏥 Healthcare ML Platform")
st.sidebar.markdown("-" * 40)

page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "📊 Data Explorer",
    "📈 EDA & Drift Analysis",
    "🤖 Model Training",
    "🔄 Cross-Temporal Evaluation",
    "🎓 Continual Learning",
    "🔍 Model Explainability",
    "💡 Final Insights"
])

# Load data
d1, d2, perf, drift, data_available = load_data()

if not data_available or d1 is None:
    # Show home page with no-data message
    if page == "🏠 Home":
        st.title("🏥 Healthcare ML Platform")
        st.markdown("-" * 80)
        st.markdown("""
        ### Production-Grade ML for Clinical Prediction
        
        This platform demonstrates a comprehensive ML system for healthcare analytics with:
        - **Temporal Distribution Shift Detection**: Identify concept drift in clinical data
        - **Cross-Temporal Model Evaluation**: Assess generalization across time periods
        - **Continual Learning**: Adapt models to new data without catastrophic forgetting
        - **Model Interpretability**: SHAP and feature importance for clinical decision support
        """)
        show_no_data_message()
    else:
        show_no_data_message()
    st.stop()

# ============== HOME PAGE ==============
if page == "🏠 Home":
    st.title("🏥 Healthcare ML Platform")
    st.markdown("-" * 80)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Dataset 1 (Historical)", f"{len(d1)} patients", "Training")
    with col2:
        st.metric("📋 Dataset 2 (Current)", f"{len(d2)} patients", "Evaluation")
    with col3:
        st.metric("👥 Total Records", f"{len(d1) + len(d2)}", "Combined")
    
    st.markdown("-" * 80)
    
    # Target distribution
    st.subheader("Target Distribution Comparison")
    target_img = os.path.join(REPORTS_DIR, "target_distribution.png")
    if os.path.exists(target_img):
        st.image(target_img, use_column_width=True)
    else:
        st.info("Target distribution plot not available yet")
    
    st.markdown("-" * 80)
    st.markdown("""
    ### About This Platform
    
    This project implements a production-grade Machine Learning pipeline for healthcare analytics,
    addressing critical challenges in real-world healthcare data:
    
    - **Temporal Distribution Shift**: Real clinical data changes over time (concept drift)
    - **Model Generalization**: Models trained on past data may not work on current data
    - **Continual Learning**: Adapt to new data patterns without retraining from scratch
    - **Interpretability**: Understand which clinical features drive predictions (SHAP)
    
    ### Key Features
    
    📊 **Data Engineering**: Feature engineering and aggregation from longitudinal EHR data  
    🔍 **Drift Detection**: Automated statistical tests for distribution shift  
    🤖 **Multi-Model Training**: Decision Trees, SVM, and Neural Networks  
    📈 **Cross-Temporal Evaluation**: Assess performance across time periods  
    🎓 **Continual Learning**: Fine-tune models on new data  
    🔍 **Explainability**: SHAP values and feature importance analysis  
    """)

# ============== DATA EXPLORER ==============
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("-" * 80)
    
    dataset_choice = st.radio(
        "Select Dataset",
        ["Dataset 1 (Historical)", "Dataset 2 (Current)"],
        horizontal=True
    )
    data = d1 if dataset_choice == "Dataset 1 (Historical)" else d2
    
    st.subheader(f"Dataset Overview - {dataset_choice}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(data)}")
    with col2:
        st.metric("Columns", f"{len(data.columns)}")
    with col3:
        st.metric("Memory", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    st.subheader("First 10 Rows")
    st.dataframe(data.head(10), use_container_width=True)
    
    st.subheader("Descriptive Statistics")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.dataframe(data[numeric_cols].describe(), use_container_width=True)
    else:
        st.info("No numeric columns found")
    
    st.subheader("Missing Values")
    missing = data.isnull().sum()
    if missing.sum() > 0:
        missing_pct = (missing / len(data)) * 100
        missing_df = pd.DataFrame({
            'Column': missing[missing > 0].index,
            'Count': missing[missing > 0].values,
            'Percentage': missing_pct[missing > 0].values
        })
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values!")

# ============== EDA PAGE ==============
elif page == "📈 EDA & Drift Analysis":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("-" * 80)
    
    st.subheader("Target Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("D1 - Healthy (0)", d1['condition_binary'].value_counts().get(0, 0))
        st.metric("D1 - Diabetes (1)", d1['condition_binary'].value_counts().get(1, 0))
    with col2:
        st.metric("D2 - Healthy (0)", d2['condition_binary'].value_counts().get(0, 0))
        st.metric("D2 - Diabetes (1)", d2['condition_binary'].value_counts().get(1, 0))
    
    st.markdown("-" * 80)
    st.subheader("Feature Distribution Analysis (Box-plots)")
    
    numeric_cols = d1.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        feature_to_plot = st.selectbox("Select Feature", numeric_cols[:10])
        box_path = os.path.join(REPORTS_DIR, f"boxplot_{feature_to_plot}.png")
        if os.path.exists(box_path):
            st.image(box_path, use_column_width=True)
        else:
            st.info(f"Box-plot for {feature_to_plot} not available")
    else:
        st.warning("No numeric features available")
    
    st.markdown("-" * 80)
    st.subheader("Temporal Drift Analysis")
    st.dataframe(drift, use_container_width=True)
    
    drift_detected = drift['drift_detected'].sum()
    st.metric("Features with Drift Detected", f"{drift_detected}/{len(drift)}")

# ============== MODEL TRAINING PAGE ==============
elif page == "🤖 Model Training":
    st.title("🤖 Model Training & Evaluation")
    st.markdown("-" * 80)
    
    st.subheader("Model Performance Summary")
    st.dataframe(perf, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy Comparison")
        if 'model' in perf.columns and 'accuracy' in perf.columns and 'dataset' in perf.columns:
            accuracy_data = perf.pivot_table(
                values='accuracy',
                index='model',
                columns='dataset',
                aggfunc='mean'
            )
            st.bar_chart(accuracy_data)
    
    with col2:
        st.subheader("F1-Score Comparison")
        if 'f1' in perf.columns:
            f1_data = perf.pivot_table(
                values='f1',
                index='model',
                columns='dataset',
                aggfunc='mean'
            )
            st.bar_chart(f1_data)

# ============== CROSS-TEMPORAL EVALUATION ==============
elif page == "🔄 Cross-Temporal Evaluation":
    st.title("🔄 Cross-Temporal Evaluation")
    st.markdown("-" * 80)
    
    st.markdown("""
    This section evaluates how models trained on historical data (Dataset 1)
    generalize to current data (Dataset 2), revealing temporal distribution shift.
    """)
    
    d1_perf = perf[perf['dataset'] == 'D1_Test']
    d2_perf = perf[perf['dataset'] == 'D2_Test']
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance on D1 (Training Distribution)")
        if len(d1_perf) > 0:
            st.dataframe(d1_perf[['model', 'accuracy', 'f1']], use_container_width=True)
    
    with col2:
        st.subheader("Performance on D2 (Test Distribution)")
        if len(d2_perf) > 0:
            st.dataframe(d2_perf[['model', 'accuracy', 'f1']], use_container_width=True)
    
    st.markdown("-" * 80)
    st.subheader("Generalization Gap Analysis")
    
    if len(d1_perf) > 0 and len(d2_perf) > 0:
        gap_data = []
        for model in d1_perf['model'].unique():
            d1_row = d1_perf[d1_perf['model'] == model]
            d2_row = d2_perf[d2_perf['model'] == model]
            
            if len(d1_row) > 0 and len(d2_row) > 0:
                d1_acc = d1_row['accuracy'].values[0]
                d2_acc = d2_row['accuracy'].values[0]
                gap = d1_acc - d2_acc
                gap_data.append({'Model': model, 'Generalization Gap': gap})
        
        gap_df = pd.DataFrame(gap_data).set_index('Model')
        st.bar_chart(gap_df)

# ============== CONTINUAL LEARNING ==============
elif page == "🎓 Continual Learning":
    st.title("🎓 Continual Learning")
    st.markdown("-" * 80)
    
    st.markdown("""
    This section demonstrates fine-tuning of the MLP model on Dataset 2 training data.
    """)
    
    mlp_d1 = perf[(perf['model'] == 'MLP') & (perf['dataset'] == 'D1_Test')]
    mlp_d2_before = perf[(perf['model'] == 'MLP') & (perf['dataset'] == 'D2_Test')]
    mlp_d2_after = perf[(perf['model'] == 'MLP_FineTuned') & (perf['dataset'] == 'D2_Test')]
    
    if len(mlp_d1) > 0 and len(mlp_d2_before) > 0 and len(mlp_d2_after) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MLP on D1", f"{mlp_d1['accuracy'].values[0]:.3f}", "Baseline")
        with col2:
            st.metric("MLP on D2 (Before CL)", f"{mlp_d2_before['accuracy'].values[0]:.3f}", "Before Fine-tuning")
        with col3:
            st.metric("MLP on D2 (After CL)", f"{mlp_d2_after['accuracy'].values[0]:.3f}", "After Fine-tuning")
        
        st.markdown("-" * 80)
        st.subheader("Continual Learning Impact")
        
        cl_comparison = pd.DataFrame({
            'Stage': ['Before CL', 'After CL'],
            'Accuracy': [mlp_d2_before['accuracy'].values[0], mlp_d2_after['accuracy'].values[0]],
            'F1-Score': [mlp_d2_before['f1'].values[0], mlp_d2_after['f1'].values[0]]
        }).set_index('Stage')
        
        st.bar_chart(cl_comparison)
        
        improvement = mlp_d2_after['accuracy'].values[0] - mlp_d2_before['accuracy'].values[0]
        if improvement > 0:
            st.success(f"✅ Accuracy Improvement: +{improvement:.4f}")
        elif improvement < 0:
            st.warning(f"⚠️ Accuracy Change: {improvement:.4f}")
    else:
        st.info("Continual learning data not available yet")

# ============== EXPLAINABILITY ==============
elif page == "🔍 Model Explainability":
    st.title("🔍 Model Explainability")
    st.markdown("-" * 80)
    
    model_choice = st.selectbox(
        "Select Model to Explain",
        ["DecisionTree", "SVM", "MLP", "MLP_FineTuned"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Feature Importance: {model_choice}")
        imp_path = os.path.join(REPORTS_DIR, f"feature_importance_{model_choice}.png")
        if os.path.exists(imp_path):
            st.image(imp_path, use_column_width=True)
        else:
            st.info("Feature importance plot not available")
    
    with col2:
        st.subheader(f"SHAP Summary: {model_choice}")
        shap_path = os.path.join(REPORTS_DIR, f"shap_heatmap_{model_choice}.png")
        if os.path.exists(shap_path):
            st.image(shap_path, use_column_width=True)
        else:
            st.info("SHAP visualization not available")

# ============== FINAL INSIGHTS ==============
elif page == "💡 Final Insights":
    st.title("💡 Final Insights & Recommendations")
    st.markdown("-" * 80)
    
    st.subheader("Key Findings")
    st.markdown("""
    ### 1. Temporal Distribution Shift
    The analysis detected significant drift in clinical features between Dataset 1 and Dataset 2,
    indicating that the clinical population characteristics have changed over time.
    
    ### 2. Model Generalization Challenges
    Models trained on historical data (D1) show performance degradation on current data (D2),
    demonstrating the impact of concept drift on model reliability.
    
    ### 3. Feature Importance
    The SHAP heatmaps and permutation importance plots reveal which clinical features
    (e.g., glucose, blood pressure) are most influential in driving predictions.
    
    ### 4. Continual Learning Effectiveness
    Fine-tuning models on new data can improve performance, showing the value of
    adapting existing models rather than retraining from scratch.
    """)
    
    st.markdown("-" * 80)
    st.subheader("Model Performance Summary")
    if len(perf) > 0:
        perf_summary = perf.groupby('model')[['accuracy', 'f1', 'precision', 'recall']].mean().round(4)
        st.dataframe(perf_summary, use_container_width=True)
    
    st.markdown("-" * 80)
    st.subheader("Recommendations for Clinical Deployment")
    st.markdown("""
    ✅ **1. Continuous Monitoring**: Implement automated drift detection in production  
    ✅ **2. Regular Retraining**: Schedule quarterly model updates on new clinical data  
    ✅ **3. Feature Validation**: Monitor top predictive features for clinical plausibility  
    ✅ **4. A/B Testing**: Compare continual learning vs. full retraining strategies  
    ✅ **5. Interpretability**: Always explain predictions to clinicians using SHAP  
    """)

st.markdown("-" * 80)
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 50px;'>
Built with ❤️ for Healthcare ML Engineering | Powered by Streamlit
</div>
""", unsafe_allow_html=True)
