"""Data preprocessing pipeline with imputation and scaling."""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

from src.config.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def get_preprocessing_pipeline():
    """Create preprocessing pipeline for numerical and categorical features.
    
    Returns:
        sklearn ColumnTransformer pipeline
    """
    # Numerical pipeline: impute missing values and scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: impute and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
    
    logger.info("Preprocessing pipeline created")
    return preprocessor
