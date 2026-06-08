import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config

# Set page config
st.set_page_config(page_title="Healthcare ML Platform", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Healthcare ML Platform")
page = st.sidebar.radio("Navigation", [
    "Home",
    "Data Explorer",
    "EDA",
    "Model Training",
    "Cross-Temporal Evaluation",
    "Continual Learning",
    "Explainability",
    "Final Insights"
])

# Load data
@st.cache_data
def load_data():
    d1 = pd.read_csv(os.path.join(config.DATA_DIR, 'processed', 'dataset1_historical.csv'))
    d2 = pd.read_csv(os.path.join(config.DATA_DIR, 'processed', 'dataset2_current.csv'))
    perf = pd.read_csv(os.path.join(config.REPORTS_DIR, 'model_performance.csv'))
    drift = pd.read_csv(os.path.join(config.REPORTS_DIR, 'drift_analysis.csv'))
    return d1, d2, perf, drift

d1, d2, perf, drift = load_data()

# ============== HOME PAGE ==============
if page == "Home":
    st.title("🏥 Healthcare ML Platform")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset 1 (Historical)", f"{len(d1)} patients", "Training")
    with col2:
        st.metric("Dataset 2 (Current)", f"{len(d2)} patients", "Evaluation")
    with col3:
        st.metric("Total Observations", f"{len(d1) + len(d2)}", "Combined")
    
    st.markdown("---")
    st.subheader("Target Distribution Comparison")
    target_img = os.path.join(config.REPORTS_DIR, "target_distribution.png")
    if os.path.exists(target_img):
        st.image(target_img, use_column_width=True)
    else:
        st.warning("Target distribution image not found")
    
    st.markdown("""
    ### Project Overview
    This platform demonstrates a production-grade ML system for healthcare analytics with:
    - **Temporal Distribution Shift Analysis**: Detect concept drift between historical and current data
    - **Cross-Temporal Generalization**: Evaluate model performance across time periods
    - **Continual Learning**: Fine-tune models on new data without catastrophic forgetting
    - **Model Interpretability**: SHAP Heatmaps and multi-model feature importance for clinical decision support
    """)

# ============== DATA EXPLORER ==============
elif page == "Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("---")
    
    dataset_choice = st.radio("Select Dataset", ["Dataset 1 (Historical)", "Dataset 2 (Current)"])
    data = d1 if dataset_choice == "Dataset 1 (Historical)" else d2
    
    st.subheader(f"Dataset Overview - {dataset_choice}")
    st.write(f"Shape: {data.shape}")
    st.dataframe(data.head(10), use_container_width=True)
    
    st.subheader("Feature Statistics")
    st.write(data[config.NUMERICAL_FEATURES].describe())
    
    st.subheader("Missing Values")
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct}).sort_values('Percentage', ascending=False)
    st.bar_chart(missing_df['Percentage'])

# ============== EDA PAGE ==============
elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("---")
    
    st.subheader("Target Distribution")
    col1, col2 = st.columns(2)
    with col1:
        target_d1 = d1['condition_binary'].value_counts()
        st.metric("Dataset 1 - Healthy (0)", target_d1.get(0, 0))
        st.metric("Dataset 1 - Diabetes (1)", target_d1.get(1, 0))
    with col2:
        target_d2 = d2['condition_binary'].value_counts()
        st.metric("Dataset 2 - Healthy (0)", target_d2.get(0, 0))
        st.metric("Dataset 2 - Diabetes (1)", target_d2.get(1, 0))
    
    st.markdown("---")
    st.subheader("Feature Distribution Analysis (Box-plots)")
    feature_to_plot = st.selectbox("Select Feature for Box-plot", config.NUMERICAL_FEATURES[:5])
    box_path = os.path.join(config.REPORTS_DIR, f"boxplot_{feature_to_plot}.png")
    if os.path.exists(box_path):
        st.image(box_path, use_column_width=True)
    else:
        st.info(f"Box-plot for {feature_to_plot} not found")

    st.markdown("---")
    st.subheader("Drift Analysis")
    st.dataframe(drift, use_container_width=True)
    
    # Display drift stats
    drift_detected = drift['drift_detected'].sum()
    st.metric("Features with Drift Detected", f"{drift_detected}/{len(drift)}")

# ============== MODEL TRAINING PAGE ==============
elif page == "Model Training":
    st.title("🤖 Model Training & Evaluation")
    st.markdown("---")
    
    st.subheader("Model Performance Summary")
    st.dataframe(perf, use_container_width=True)
    
    # Performance comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accuracy Comparison")
        accuracy_data = perf.pivot_table(values='accuracy', index='model', columns='dataset')
        st.bar_chart(accuracy_data)
    
    with col2:
        st.subheader("F1-Score Comparison")
        f1_data = perf.pivot_table(values='f1', index='model', columns='dataset')
        st.bar_chart(f1_data)

# ============== CROSS-TEMPORAL EVALUATION ==============
elif page == "Cross-Temporal Evaluation":
    st.title("🔄 Cross-Temporal Evaluation")
    st.markdown("---")
    
    st.subheader("Temporal Generalization Analysis")
    st.write("""
    This section evaluates how models trained on historical data (Dataset 1) 
    generalize to current data (Dataset 2), revealing temporal distribution shift.
    """)
    
    # Extract D1 and D2 performance
    d1_perf = perf[perf['dataset'] == 'D1_Test']
    d2_perf = perf[perf['dataset'] == 'D2_Test']
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance on D1 (Training Distribution)")
        st.dataframe(d1_perf[['model', 'accuracy', 'f1']], use_container_width=True)
    
    with col2:
        st.subheader("Performance on D2 (Test Distribution)")
        st.dataframe(d2_perf[['model', 'accuracy', 'f1']], use_container_width=True)
    
    st.markdown("---")
    st.subheader("Generalization Gap Analysis")
    
    # Calculate generalization gap
    gap_data = []
    for model in d1_perf['model'].unique():
        d1_acc = d1_perf[d1_perf['model'] == model]['accuracy'].values[0]
        d2_acc = d2_perf[d2_perf['model'] == model]['accuracy'].values[0]
        gap = d1_acc - d2_acc
        gap_data.append({'Model': model, 'Generalization Gap': gap})
    
    gap_df = pd.DataFrame(gap_data).set_index('Model')
    st.bar_chart(gap_df)

# ============== CONTINUAL LEARNING ==============
elif page == "Continual Learning":
    st.title("🔄 Continual Learning")
    st.markdown("---")
    
    st.write("""
    This section demonstrates fine-tuning of the MLP model on Dataset 2 training data.
    """)
    
    # Extract MLP performance before and after CL
    mlp_d1 = perf[(perf['model'] == 'MLP') & (perf['dataset'] == 'D1_Test')]
    mlp_d2_before = perf[(perf['model'] == 'MLP') & (perf['dataset'] == 'D2_Test')]
    mlp_d2_after = perf[(perf['model'] == 'MLP_FineTuned') & (perf['dataset'] == 'D2_Test')]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MLP on D1", f"{mlp_d1['accuracy'].values[0]:.3f}", "Baseline")
    with col2:
        st.metric("MLP on D2 (Before CL)", f"{mlp_d2_before['accuracy'].values[0]:.3f}", "Before Fine-tuning")
    with col3:
        st.metric("MLP on D2 (After CL)", f"{mlp_d2_after['accuracy'].values[0]:.3f}", "After Fine-tuning")
    
    st.markdown("---")
    st.subheader("Continual Learning Impact")
    
    cl_comparison = pd.DataFrame({
        'Stage': ['Before CL', 'After CL'],
        'Accuracy': [mlp_d2_before['accuracy'].values[0], mlp_d2_after['accuracy'].values[0]],
        'F1-Score': [mlp_d2_before['f1'].values[0], mlp_d2_after['f1'].values[0]]
    }).set_index('Stage')
    
    st.bar_chart(cl_comparison)
    
    improvement = mlp_d2_after['accuracy'].values[0] - mlp_d2_before['accuracy'].values[0]
    st.success(f"✅ Accuracy Improvement: +{improvement:.4f}")

# ============== EXPLAINABILITY ==============
elif page == "Explainability":
    st.title("🔍 Model Explainability")
    st.markdown("---")
    
    model_choice = st.selectbox("Select Model to Explain", ["DecisionTree", "SVM", "MLP", "MLP_FineTuned"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Feature Importance: {model_choice}")
        imp_path = os.path.join(config.REPORTS_DIR, f"feature_importance_{model_choice}.png")
        if os.path.exists(imp_path):
            st.image(imp_path, use_column_width=True)
        else:
            st.info("Importance plot not available")
            
    with col2:
        st.subheader(f"SHAP Heatmap/Summary: {model_choice}")
        shap_path = os.path.join(config.REPORTS_DIR, f"shap_heatmap_{model_choice}.png")
        if os.path.exists(shap_path):
            st.image(shap_path, use_column_width=True)
        else:
            st.info("SHAP visualization not available")

# ============== FINAL INSIGHTS ==============
elif page == "Final Insights":
    st.title("💡 Final Insights & Recommendations")
    st.markdown("---")
    
    st.subheader("Key Findings (Real Dataset)")
    st.write("""
    ### 1. Temporal Distribution Shift
    - Detected significant drift in real-world clinical features between Dataset 1 and Dataset 2.
    - Features like Systolic BP and Glucose show variations across the 2015 temporal split.
    
    ### 2. Model Generalization & Explainability
    - The **SHAP Heatmaps** reveal how different models utilize clinical features across patient samples.
    - **Feature Importance** across all 4 models (DT, SVM, MLP, and Fine-tuned MLP) shows consistency in clinical drivers like Glucose and Blood Pressure.
    
    ### 3. Continual Learning Effectiveness
    - Fine-tuning the MLP on Dataset 2 data shows how models can adapt to temporal shifts.
    
    ### 4. Clinical Implications
    - Continuous monitoring of feature drift using **Box-plots** and **KS Tests** is essential for maintaining clinical trust.
    """)
    
    st.markdown("---")
    st.subheader("Model Performance Summary")
    st.dataframe(perf.groupby('model')[['accuracy', 'f1', 'precision', 'recall']].mean().round(4))
    
    st.markdown("---")
    st.subheader("Recommendations")
    st.write("""
    ✅ **1. Continuous Monitoring**: Implement drift detection pipeline in production
    ✅ **2. Model Retraining**: Schedule quarterly fine-tuning on new data
    ✅ **3. Feature Validation**: Validate top features regularly (Glucose, BP)
    ✅ **4. A/B Testing**: Test continual learning vs retraining from scratch
    """)

st.markdown("---")
st.markdown("Built with ❤️ for Healthcare ML Engineering | Streamlit Cloud Deployment")
