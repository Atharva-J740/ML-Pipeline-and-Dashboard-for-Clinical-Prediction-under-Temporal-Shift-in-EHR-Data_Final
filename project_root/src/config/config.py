"""Configuration settings for Healthcare ML Pipeline."""
import os
from pathlib import Path

# Get base directory dynamically
BASE_DIR = Path(__file__).parent.parent.parent.absolute()

# Paths
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw' / 'real_dataset' / 'DATA'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models' / 'saved_models'
REPORTS_DIR = BASE_DIR / 'reports'

# Ensure directories exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

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

# Convert Path objects to strings for sklearn compatibility
RAW_DATA_DIR = str(RAW_DATA_DIR)
PROCESSED_DATA_DIR = str(PROCESSED_DATA_DIR)
MODELS_DIR = str(MODELS_DIR)
REPORTS_DIR = str(REPORTS_DIR)
BASE_DIR = str(BASE_DIR)
