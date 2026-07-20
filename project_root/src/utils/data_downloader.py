"""Google Drive data downloader utility for EHR datasets."""
import os
import gdown
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Google Drive folder ID from the shared link
GDRIVE_FOLDER_ID = "1dPkA16Cux6zOCpDz32fLY8V66UNKMtB8"
GDRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"


class DataDownloader:
    """Downloads EHR CSV files from Google Drive."""
    
    def __init__(self, output_dir: str):
        """Initialize downloader with output directory.
        
        Args:
            output_dir: Path where CSV files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_datasets(self) -> bool:
        """Download all EHR datasets from Google Drive.
        
        Returns:
            True if all files downloaded successfully, False otherwise
        """
        try:
            logger.info(f"Downloading datasets from Google Drive: {GDRIVE_FOLDER_URL}")
            
            # Download entire folder
            gdown.download_folder(
                url=GDRIVE_FOLDER_URL,
                output=str(self.output_dir),
                quiet=False,
                use_cookies=False
            )
            
            logger.info(f"Datasets downloaded to {self.output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading datasets: {e}")
            return False
    
    def verify_datasets(self, required_files: list = None) -> bool:
        """Verify that all required CSV files exist.
        
        Args:
            required_files: List of required filenames. Defaults to standard EHR files.
            
        Returns:
            True if all required files exist, False otherwise
        """
        if required_files is None:
            required_files = ['patients.csv', 'observations.csv', 'conditions.csv']
        
        missing = []
        for filename in required_files:
            filepath = self.output_dir / filename
            if not filepath.exists():
                missing.append(filename)
        
        if missing:
            logger.warning(f"Missing files: {missing}")
            return False
        
        logger.info(f"All required files found in {self.output_dir}")
        return True
    
    def get_file_path(self, filename: str) -> str:
        """Get full path to a dataset file.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Full path to the file
        """
        return str(self.output_dir / filename)
