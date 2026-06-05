# Healthcare ML Platform: Temporal Drift, Continual Learning, and Explainability

## Overview

This project implements a production-grade Machine Learning (ML) pipeline for healthcare analytics, focusing on clinical prediction tasks. It addresses critical challenges in real-world healthcare data, such as **temporal distribution shift** (concept drift), **model generalization**, and the need for **continual learning** and **model interpretability**. The platform is designed to be modular, robust, and easily deployable across various development environments.

## Key Features

*   **Data Engineering**: Robust ingestion, merging, and advanced feature engineering for Electronic Health Record (EHR) data from a real-world dataset.
*   **Temporal Data Splitting**: Divides data into historical (Dataset 1) and current (Dataset 2) periods to simulate real-world data drift.
*   **Exploratory Data Analysis (EDA)**: Comprehensive descriptive statistics, target distribution analysis, and feature distribution comparisons using **box-plots**.
*   **Drift Analysis**: Automated detection of feature drift between historical and current datasets using statistical tests (e.g., Kolmogorov-Smirnov).
*   **Model Training**: Implements and trains various classification models, including Decision Trees, Support Vector Machines (SVM), and Multi-Layer Perceptrons (MLP).
*   **Cross-Temporal Evaluation**: Evaluates model performance on both historical and current test sets to assess generalization capabilities over time.
*   **Continual Learning**: Demonstrates fine-tuning of models on new data to adapt to distribution shifts, preventing catastrophic forgetting.
*   **Model Explainability**: Provides **SHAP (SHapley Additive exPlanations) heatmaps** and **multi-model feature importance plots** for transparent clinical decision support across Decision Tree, SVM, MLP, and Fine-tuned MLP models.
*   **Interactive Dashboard**: A multi-page Streamlit application for interactive viewing of data, model performance, drift analysis, and explainability insights.

## Project Structure

```
project_root/
├── data/
│   ├── raw/                 # Raw downloaded datasets
│   │   └── real_dataset/    # Real EHR dataset from Google Drive
│   │       └── DATA/        # CSV files of the real dataset
│   ├── processed/           # Processed D1 and D2 datasets
│   └── interim/             # Intermediate data files
├── models/
│   ├── saved_models/        # Trained model pipelines
│   └── continual_learning/  # Fine-tuned models
├── reports/                 # Generated plots, CSV reports, and visualizations
├── src/
│   ├── config/              # Configuration settings (config.py)
│   ├── data_ingestion/      # Data loading and initial feature engineering
│   ├── temporal_split/      # Logic for splitting data into D1 and D2
│   ├── preprocessing/       # Data preprocessing pipelines
│   ├── eda/                 # Exploratory Data Analysis and drift detection
│   ├── training/            # Model training utilities
│   ├── evaluation/          # Model evaluation metrics and plots
│   ├── continual_learning/  # Continual learning strategies
│   ├── explainability/      # SHAP and feature importance modules
│   └── utils/               # Utility functions (e.g., data_generator.py - now disabled)
├── dashboard/               # Streamlit application files
│   └── dashboard.py         # Main Streamlit app
├── main.py                  # Main script to run the entire ML pipeline
├── requirements.txt         # Python dependencies
└── README.md                # Project README (this file)
```

## Setup and Installation

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### 1. Clone the Repository

```bash
git clone <repository_url> # Replace <repository_url> with the actual URL
cd project_root
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Obtain the Real Dataset

The project uses a real EHR dataset from Google Drive. You need to download these files and place them in the correct directory.

1.  **Download the Dataset**: Access the Google Drive folder: [https://drive.google.com/drive/folders/1dPkA16Cux6zOCpDz32fLY8V66UNKMtB8](https://drive.google.com/drive/folders/1dPkA16Cux6zOCpDz32fLY8V66UNKMtB8)
2.  **Create Directory**: Inside your `project_root/data/raw/` directory, create a new folder named `real_dataset/DATA/`.
    ```bash
mkdir -p data/raw/real_dataset/DATA
    ```
3.  **Place Files**: Download all `.csv` files from the Google Drive folder and place them directly into the `project_root/data/raw/real_dataset/DATA/` directory.

    *Alternatively, you can use `gdown` if you have it installed and permissions are set correctly:*
    ```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1dPkA16Cux6zOCpDz32fLY8V66UNKMtB8 -O data/raw/real_dataset/
    ```

## Running the ML Pipeline

Once the data is in place and dependencies are installed, you can run the entire ML pipeline from the `project_root` directory:

```bash
python3 main.py
```

This script will:
*   Ingest and preprocess the real dataset.
*   Perform temporal splitting into Dataset 1 (Historical) and Dataset 2 (Current).
*   Conduct EDA and drift analysis, saving plots and reports to `reports/`.
*   Train Decision Tree, SVM, and MLP models.
*   Evaluate models on both D1 and D2 test sets, saving performance metrics to `reports/model_performance.csv`.
*   Perform continual learning on the MLP model.
*   Generate SHAP heatmaps and feature importance plots for all models, saving them to `reports/`.

## Running the Streamlit Dashboard

After the pipeline has run and generated all necessary reports and models, you can launch the interactive dashboard:

```bash
cd project_root
streamlit run dashboard/dashboard.py
```

Streamlit will typically open a new tab in your web browser with the dashboard. If not, it will provide a local URL (e.g., `http://localhost:8501`) and a network URL that you can use to access it.

## Environment Specific Instructions

### VS Code

1.  **Open Folder**: Open the `project_root` folder in VS Code (`File > Open Folder...`).
2.  **Select Python Interpreter**: VS Code should detect your `venv` virtual environment. If not, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), search for "Python: Select Interpreter", and choose the one inside your `venv` folder.
3.  **Run `main.py`**: You can run `main.py` by clicking the "Run Python File" button in the top right corner or by right-clicking the file in the Explorer and selecting "Run Python File in Terminal".
4.  **Run Streamlit**: Open a new terminal in VS Code (`Terminal > New Terminal`), navigate to `project_root`, and run `streamlit run dashboard/dashboard.py`.

### Google Colab

1.  **Upload Project**: Upload the entire `project_root` folder to your Google Drive. Then, in a Colab notebook, mount your Google Drive:
    ```python
    from google.colab import drive
    drive.mount("/content/drive")
    ```
2.  **Navigate**: Change your current directory to the `project_root` within your mounted drive:
    ```python
    %cd /content/drive/MyDrive/path/to/your/project_root
    ```
3.  **Install Dependencies**: 
    ```bash
    !pip install -r requirements.txt
    ```
4.  **Download Data**: Use `gdown` directly in Colab to download the dataset:
    ```python
    !pip install gdown
    !mkdir -p data/raw/real_dataset/DATA
    !gdown --folder https://drive.google.com/drive/folders/1dPkA16Cux6zOCpDz32fLY8V66UNKMtB8 -O data/raw/real_dataset/
    ```
5.  **Run Pipeline**: 
    ```python
    !python main.py
    ```
6.  **Run Streamlit**: For Streamlit in Colab, you'll need `ngrok` to expose the local server. 
    ```bash
    !pip install streamlit_ngrok
    from streamlit_ngrok import streamlit_run
    
    # In a separate cell or after the above setup
    streamlit_run("dashboard/dashboard.py")
    ```
    This will provide a public URL to access your dashboard.

### Jupyter Notebook

1.  **Navigate**: Open your Jupyter Notebook/Lab and navigate to the `project_root` directory.
2.  **Install Dependencies**: Open a terminal within Jupyter (or your system terminal) and install dependencies:
    ```bash
pip install -r requirements.txt
    ```
3.  **Download Data**: Ensure the dataset is downloaded as described in "Obtain the Real Dataset" section.
4.  **Run Pipeline**: In a notebook cell, execute:
    ```python
    %run main.py
    ```
5.  **Run Streamlit**: Open a new terminal within Jupyter (or your system terminal) and run:
    ```bash
    streamlit run dashboard/dashboard.py
    ```
    Access the dashboard via the provided local URL.

## Troubleshooting

*   **`gdown` 401 Error**: Ensure the Google Drive folder has "Anyone with the link" access. If not, update permissions or manually download the files.
*   **`ParserError` when loading CSVs**: Real-world CSVs can be messy. The current `DataIngestor` uses `on_bad_lines='skip'` to handle minor issues. For severe errors, inspect the problematic CSV file manually.
*   **`ValueError: Could not interpret value...` in Streamlit**: This often means a column expected by a plot (e.g., `condition`) is missing or renamed. Verify `config.py` and `ingestor.py` correctly map features.
*   **Streamlit Dashboard Not Loading**: Check the terminal where Streamlit was launched for error messages. Ensure port 8501 is not blocked by a firewall or another application.
*   **`shap` or `permutation_importance` errors**: Ensure you have the latest versions of `shap` and `scikit-learn`. Some functions might have changed parameter names (e.g., `random_seed` to `random_state`). The current `explainer.py` handles this.

---

Built with ❤️ for Healthcare ML Engineering | Manus Platform
