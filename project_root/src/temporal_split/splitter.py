"""Temporal data splitting into historical and current datasets."""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from src.config.config import SPLIT_DATE


class TemporalSplitter:
    """Splits data into historical (D1) and current (D2) datasets based on temporal cutoff."""
    
    def __init__(self, split_date=SPLIT_DATE):
        """Initialize splitter with cutoff date.
        
        Args:
            split_date: Date string (YYYY-MM-DD) to split historical vs current data
        """
        self.split_date = pd.to_datetime(split_date)
        logger.info(f"TemporalSplitter initialized with split_date: {self.split_date}")
        
    def split(self, df):
        """Split dataframe into historical and current datasets.
        
        Args:
            df: Input dataframe with 'diagnosis_date' column
            
        Returns:
            Tuple of (dataset1_historical, dataset2_current)
        """
        logger.info(f"Splitting data at {self.split_date}...")
        
        # Ensure diagnosis_date is datetime
        df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
        
        dataset1 = df[df['diagnosis_date'] <= self.split_date].copy()
        dataset2 = df[df['diagnosis_date'] > self.split_date].copy()
        
        logger.info(f"Dataset 1 (Historical) size: {len(dataset1)}")
        logger.info(f"Dataset 2 (Current) size: {len(dataset2)}")
        
        return dataset1, dataset2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_ingestion.ingestor import DataIngestor, FeatureEngineer
    
    ingestor = DataIngestor()
    patients, obs, cond = ingestor.load_data()
    
    fe = FeatureEngineer()
    obs_agg = fe.aggregate_observations(obs)
    merged_df = fe.merge_datasets(patients, obs_agg, cond)
    
    splitter = TemporalSplitter()
    d1, d2 = splitter.split(merged_df)
    
    print(f"\nDataset 1 (Historical) size: {len(d1)}")
    print(f"Dataset 2 (Current) size: {len(d2)}")
