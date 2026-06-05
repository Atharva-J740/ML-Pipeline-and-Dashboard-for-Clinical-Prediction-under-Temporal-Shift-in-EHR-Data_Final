import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import config

class TemporalSplitter:
    def __init__(self, split_date=config.SPLIT_DATE):
        self.split_date = pd.to_datetime(split_date)
        
    def split(self, df):
        # We use 'diagnosis_date' as the temporal reference
        df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
        
        # Ensure diagnosis_date is datetime
        df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
        dataset1 = df[df['diagnosis_date'] <= self.split_date].copy()
        dataset2 = df[df['diagnosis_date'] > self.split_date].copy()
        
        return dataset1, dataset2

if __name__ == "__main__":
    from src.data_ingestion.ingestor import DataIngestor, FeatureEngineer
    
    ingestor = DataIngestor()
    demo, obs, cond = ingestor.load_data()
    fe = FeatureEngineer()
    obs_agg = fe.aggregate_observations(obs)
    merged_df = fe.merge_datasets(demo, obs_agg, cond)
    
    splitter = TemporalSplitter()
    d1, d2 = splitter.split(merged_df)
    
    print(f"Dataset 1 (Historical) size: {len(d1)}")
    print(f"Dataset 2 (Current) size: {len(d2)}")
