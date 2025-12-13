"""
Data Loading Utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and validating data"""
    
    def __init__(self, data_dir='data/raw'):
        """
        Initialize DataLoader
        
        Parameters:
        -----------
        data_dir : str or Path
            Path to the data directory relative to project root
        """
        # Get project root (assuming src/ is in project root)
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / data_dir
        
        logger.info(f"DataLoader initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Data directory: {self.data_dir}")
        
    def load_xente_data(self, sample_size=None):
        """
        Load Xente dataset
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of rows to sample (for testing)
            
        Returns:
        --------
        pandas DataFrame: Loaded dataset
        """
        try:
            data_path = self.data_dir / 'data.csv'
            
            logger.info(f"Loading data from: {data_path}")
            logger.info(f"File exists: {data_path.exists()}")
            
            if not data_path.exists():
                # Try alternative paths
                logger.warning(f"File not found at primary location. Searching...")
                possible_paths = [
                    self.project_root / 'data' / 'raw' / 'data.csv',
                    self.project_root / 'data.csv',
                    Path.cwd() / 'data' / 'raw' / 'data.csv',
                    Path.cwd() / 'data.csv'
                ]
                
                for path in possible_paths:
                    if path.exists():
                        data_path = path
                        logger.info(f"Found data at: {data_path}")
                        break
            
            # Load main dataset
            if sample_size:
                df = pd.read_csv(data_path, nrows=sample_size)
                logger.info(f"Loaded sample of {sample_size} rows")
            else:
                df = pd.read_csv(data_path)
                logger.info(f"Loaded full dataset: {len(df):,} rows × {len(df.columns)} columns")
            
            # Basic validation
            self._validate_data(df)
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Data file not found at any location")
            logger.error("Please ensure data.csv exists in data/raw/ folder")
            
            # List available files
            if self.data_dir.exists():
                logger.info(f"Files in {self.data_dir}:")
                for file in self.data_dir.iterdir():
                    logger.info(f"  - {file.name}")
            raise
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_data(self, df):
        """Perform basic data validation"""
        logger.info("Starting data validation...")
        
        required_columns = [
            'TransactionId', 'CustomerId', 'Amount', 
            'TransactionStartTime', 'FraudResult'
        ]
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")
        else:
            logger.info("✅ All required columns present")
        
        # Dataset statistics
        logger.info(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Check for negative amounts (should be credits)
        if 'Amount' in df.columns:
            negative_amounts = (df['Amount'] < 0).sum()
            if negative_amounts > 0:
                logger.info(f"Found {negative_amounts:,} credit transactions (negative amounts)")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates:,} duplicate rows ({duplicates/len(df)*100:.2f}%)")
        else:
            logger.info("✅ No duplicate rows found")
        
        # Check for missing values
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            missing_cols = df.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            logger.warning(f"Found {missing_total:,} missing values in {len(missing_cols)} columns")
            for col, count in missing_cols.items():
                logger.info(f"  - {col}: {count:,} missing ({count/len(df)*100:.2f}%)")
        else:
            logger.info("✅ No missing values found")
        
        logger.info("Data validation completed")
    
    def save_processed_data(self, df, filename, directory='data/processed'):
        """
        Save processed data to file
        
        Parameters:
        -----------
        df : pandas DataFrame
            Data to save
        filename : str
            Output filename
        directory : str
            Directory to save to (relative to project root)
        """
        output_dir = self.project_root / directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        # Use parquet for efficiency, fall back to CSV if needed
        try:
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved processed data to {output_path} (Parquet format)")
        except:
            df.to_csv(output_path.with_suffix('.csv'), index=False)
            logger.info(f"Saved processed data to {output_path.with_suffix('.csv')} (CSV format)")