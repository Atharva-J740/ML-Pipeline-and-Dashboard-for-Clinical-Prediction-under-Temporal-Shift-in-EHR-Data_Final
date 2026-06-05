import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw', 'real_dataset', 'DATA')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Data Split Config
SPLIT_DATE = '2015-01-01'  # Date to separate Dataset 1 (Historical) and Dataset 2 (Current)

# Model Config
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Target Variable
TARGET_COLUMN = 'condition_binary'

# Feature Selection
NUMERICAL_FEATURES = [
    'age', 'systolic_blood_pressure_mean', 'systolic_blood_pressure_std', 
    'diastolic_blood_pressure_mean', 'heart_rate_mean', 'body_temperature_mean', 
    'glucose_mass_volume_in_blood_mean', 'cholesterol_mass_volume_in_serum_or_plasma_mean'
]
CATEGORICAL_FEATURES = ['GENDER', 'RACE', 'ETHNICITY']
