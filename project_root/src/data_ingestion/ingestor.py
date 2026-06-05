import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import config

class DataIngestor:
    def __init__(self, raw_data_dir=config.RAW_DATA_DIR):
        self.raw_data_dir = raw_data_dir
        
    def load_data(self):
        # Load real dataset files
        # Using on_bad_lines='skip' to handle parsing issues in real-world EHR data
        patients = pd.read_csv(os.path.join(self.raw_data_dir, 'patients.csv'), on_bad_lines='skip')
        observations = pd.read_csv(os.path.join(self.raw_data_dir, 'observations.csv'), on_bad_lines='skip', parse_dates=['DATE'])
        conditions = pd.read_csv(os.path.join(self.raw_data_dir, 'conditions.csv'), on_bad_lines='skip', parse_dates=['START'])
        
        # Rename 'Id' to 'patient_id' for consistency if needed, but we'll use 'PATIENT' and 'Id'
        patients.rename(columns={'Id': 'PATIENT_ID'}, inplace=True)
        observations.rename(columns={'PATIENT': 'PATIENT_ID', 'DATE': 'timestamp'}, inplace=True)
        conditions.rename(columns={'PATIENT': 'PATIENT_ID', 'START': 'diagnosis_date'}, inplace=True)
        
        return patients, observations, conditions

class FeatureEngineer:
    def __init__(self):
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
        # Filter for relevant observations
        obs_filtered = observations[observations['DESCRIPTION'].isin(self.desc_mapping.keys())].copy()
        obs_filtered['feature_name'] = obs_filtered['DESCRIPTION'].map(self.desc_mapping)
        
        # Ensure VALUE is numeric
        obs_filtered['VALUE'] = pd.to_numeric(obs_filtered['VALUE'], errors='coerce')
        obs_filtered.dropna(subset=['VALUE'], inplace=True)
        
        # Aggregate longitudinal observations per patient and feature
        obs_agg = obs_filtered.groupby(['PATIENT_ID', 'feature_name'])['VALUE'].agg(['mean', 'std', 'max', 'min']).unstack()
        
        # Flatten multi-index columns
        obs_agg.columns = [f"{feat}_{stat}" for stat, feat in obs_agg.columns]
        obs_agg.reset_index(inplace=True)
        return obs_agg
    
    def merge_datasets(self, patients, obs_agg, conditions):
        # Merge patients with aggregated observations
        df = patients.merge(obs_agg, on='PATIENT_ID', how='inner')
        
        # Calculate age from BIRTHDATE
        df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'])
        df['age'] = (pd.to_datetime('today').year - df['BIRTHDATE'].dt.year)
        
        # Identify Diabetes patients in conditions
        diabetes_ids = conditions[conditions['DESCRIPTION'].str.contains('diabetes', case=False, na=False)]['PATIENT_ID'].unique()
        df['condition_binary'] = df['PATIENT_ID'].isin(diabetes_ids).astype(int)
        
        # Add diagnosis_date for temporal splitting
        # For patients with the condition, use the earliest diagnosis date
        # For healthy patients, use a placeholder or the last observation date
        diag_dates = conditions.groupby('PATIENT_ID')['diagnosis_date'].min().reset_index()
        df = df.merge(diag_dates, on='PATIENT_ID', how='left')
        
        # Fill missing diagnosis dates for healthy patients with a late date to keep them in D2 or an early one for D1
        # Here we use the last observation date as a reference for splitting
        df['diagnosis_date'] = df['diagnosis_date'].fillna(pd.to_datetime('2020-01-01'))
        
        return df

if __name__ == "__main__":
    ingestor = DataIngestor()
    patients, obs, cond = ingestor.load_data()
    
    fe = FeatureEngineer()
    obs_agg = fe.aggregate_observations(obs)
    merged_df = fe.merge_datasets(patients, obs_agg, cond)
    
    print(f"Merged dataset shape: {merged_df.shape}")
    print(merged_df.head())
