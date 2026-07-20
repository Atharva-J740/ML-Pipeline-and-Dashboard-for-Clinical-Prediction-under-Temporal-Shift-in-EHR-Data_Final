"""Data ingestion and feature engineering for EHR datasets."""
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Import config without sys.path manipulation
try:
    from src.config.config import (
        RAW_DATA_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
    )
except ImportError:
    # Fallback for direct script execution
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.config.config import (
        RAW_DATA_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
    )


class DataIngestor:
    """Loads and validates EHR data from CSV files."""
    
    def __init__(self, raw_data_dir=None):
        """Initialize DataIngestor with data directory.
        
        Args:
            raw_data_dir: Path to directory containing CSV files. 
                         If None, attempts automatic download from Google Drive.
        """
        self.raw_data_dir = raw_data_dir or RAW_DATA_DIR
        self._ensure_data_available()
        
    def _ensure_data_available(self):
        """Ensure data files exist, download if necessary."""
        required_files = ['patients.csv', 'observations.csv', 'conditions.csv']
        data_path = Path(self.raw_data_dir)
        
        # Check if all files exist
        existing_files = [f for f in required_files if (data_path / f).exists()]
        missing_files = [f for f in required_files if f not in existing_files]
        
        if missing_files:
            logger.info(f"Missing files: {missing_files}. Attempting download...")
            self._download_from_gdrive()
    
    def _download_from_gdrive(self):
        """Download datasets from Google Drive using gdown."""
        try:
            from src.utils.data_downloader import DataDownloader
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from src.utils.data_downloader import DataDownloader
        
        downloader = DataDownloader(self.raw_data_dir)
        
        if not downloader.verify_datasets():
            logger.warning("Required datasets not found. Downloading from Google Drive...")
            if downloader.download_datasets():
                logger.info("Download completed successfully.")
                if downloader.verify_datasets():
                    logger.info("All datasets verified.")
                    return
            logger.error("Failed to download datasets from Google Drive.")
            logger.error(f"Please ensure data files are placed in: {self.raw_data_dir}")
        
    def load_data(self):
        """Load patients, observations, and conditions data.
        
        Returns:
            Tuple of (patients_df, observations_df, conditions_df)
            
        Raises:
            FileNotFoundError: If required CSV files not found
        """
        try:
            logger.info(f"Loading data from {self.raw_data_dir}")
            
            # Load CSV files with robust error handling
            patients = pd.read_csv(
                os.path.join(self.raw_data_dir, 'patients.csv'),
                on_bad_lines='skip'
            )
            logger.info(f"Loaded patients.csv: {patients.shape}")
            
            observations = pd.read_csv(
                os.path.join(self.raw_data_dir, 'observations.csv'),
                on_bad_lines='skip',
                parse_dates=['DATE']
            )
            logger.info(f"Loaded observations.csv: {observations.shape}")
            
            conditions = pd.read_csv(
                os.path.join(self.raw_data_dir, 'conditions.csv'),
                on_bad_lines='skip',
                parse_dates=['START']
            )
            logger.info(f"Loaded conditions.csv: {conditions.shape}")
            
            # Standardize column names
            patients.rename(columns={'Id': 'PATIENT_ID'}, inplace=True)
            observations.rename(
                columns={'PATIENT': 'PATIENT_ID', 'DATE': 'timestamp'},
                inplace=True
            )
            conditions.rename(
                columns={'PATIENT': 'PATIENT_ID', 'START': 'diagnosis_date'},
                inplace=True
            )
            
            return patients, observations, conditions
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            logger.error(f"Expected files in: {self.raw_data_dir}")
            logger.error("Please download data from Google Drive or place CSV files manually.")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise


class FeatureEngineer:
    """Aggregates observations and merges datasets for feature engineering."""
    
    def __init__(self):
        """Initialize with observation description mappings."""
        # Mapping real descriptions to normalized feature names
        self.desc_mapping = {
            'Systolic Blood Pressure': 'systolic_blood_pressure',
            'Diastolic Blood Pressure': 'diastolic_blood_pressure',
            'Heart rate': 'heart_rate',
            'Body temperature': 'body_temperature',
            'Glucose [Mass/volume] in Blood': 'glucose_mass_volume_in_blood',
            'Cholesterol [Mass/volume] in Serum or Plasma': 'cholesterol_mass_volume_in_serum_or_plasma'
        }
        
    def aggregate_observations(self, observations):
        """Aggregate longitudinal observations per patient.
        
        Args:
            observations: DataFrame with patient observation records
            
        Returns:
            DataFrame with aggregated statistics (mean, std, max, min) per feature
        """
        logger.info("Aggregating observations...")
        
        # Filter for relevant observations
        obs_filtered = observations[
            observations['DESCRIPTION'].isin(self.desc_mapping.keys())
        ].copy()
        
        obs_filtered['feature_name'] = obs_filtered['DESCRIPTION'].map(self.desc_mapping)
        
        # Ensure VALUE is numeric
        obs_filtered['VALUE'] = pd.to_numeric(obs_filtered['VALUE'], errors='coerce')
        obs_filtered.dropna(subset=['VALUE'], inplace=True)
        
        logger.info(f"Filtered observations: {obs_filtered.shape}")
        
        # Aggregate longitudinal observations per patient and feature
        obs_agg = obs_filtered.groupby(['PATIENT_ID', 'feature_name'])['VALUE'].agg(
            ['mean', 'std', 'max', 'min']
        ).unstack()
        
        # Flatten multi-index columns
        obs_agg.columns = [f"{feat}_{stat}" for stat, feat in obs_agg.columns]
        obs_agg.reset_index(inplace=True)
        
        logger.info(f"Aggregated observations shape: {obs_agg.shape}")
        return obs_agg
    
    def merge_datasets(self, patients, obs_agg, conditions):
        """Merge patients, aggregated observations, and conditions.
        
        Args:
            patients: Patient demographic data
            obs_agg: Aggregated observations
            conditions: Patient conditions/diagnoses
            
        Returns:
            Merged DataFrame with engineered features
        """
        logger.info("Merging datasets...")
        
        # Merge patients with aggregated observations
        df = patients.merge(obs_agg, on='PATIENT_ID', how='inner')
        logger.info(f"After merging observations: {df.shape}")
        
        # Calculate age from BIRTHDATE
        df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'])
        df['age'] = (pd.to_datetime('today').year - df['BIRTHDATE'].dt.year)
        
        # Identify Diabetes patients
        diabetes_ids = conditions[
            conditions['DESCRIPTION'].str.contains('diabetes', case=False, na=False)
        ]['PATIENT_ID'].unique()
        df['condition_binary'] = df['PATIENT_ID'].isin(diabetes_ids).astype(int)
        
        # Add diagnosis_date for temporal splitting
        diag_dates = conditions.groupby('PATIENT_ID')['diagnosis_date'].min().reset_index()
        df = df.merge(diag_dates, on='PATIENT_ID', how='left')
        
        # Fill missing diagnosis dates for healthy patients
        df['diagnosis_date'] = df['diagnosis_date'].fillna(pd.to_datetime('2020-01-01'))
        
        logger.info(f"Final merged dataset shape: {df.shape}")
        logger.info(f"Target distribution:\n{df['condition_binary'].value_counts()}")
        
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    ingestor = DataIngestor()
    patients, obs, cond = ingestor.load_data()
    
    fe = FeatureEngineer()
    obs_agg = fe.aggregate_observations(obs)
    merged_df = fe.merge_datasets(patients, obs_agg, cond)
    
    print(f"\nMerged dataset shape: {merged_df.shape}")
    print(f"\nColumns: {merged_df.columns.tolist()}")
    print(f"\nFirst few rows:\n{merged_df.head()}")
