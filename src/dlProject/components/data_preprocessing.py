import time
from pathlib import Path

from dlProject import logger
from dlProject.utils.common import read_file
from dlProject.utils.features import *
from dlProject.entity.config_entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
    
    def preprocess_data(self):
        start_time = time.time()
        file = Path(self.config.data_source_dir, self.config.data_file_name)
        df = read_file(file)
        
        # Create datasets based on data_source values
        d1 = df[df['data_source'].isin(data_sources_solana)].copy()
        d2 = df[df['data_source'].isin(data_sources_homeoffice)].copy()
        d3 = df[df['data_source'].isin(data_sources_solanatest)].copy()

        # Print the shapes of each dataset to verify
        logger.info(f"Shape of d1 (Solana dataset): {d1.shape}")
        logger.info(f"Shape of d2 (SolanaTest dataset): {d2.shape}") 
        logger.info(f"Shape of d3 (Solana Home Office dataset): {d3.shape}")
        
        # Save the datasets to parquet files
        d1.to_csv(Path(self.config.root_dir, 'data_sources_solana.csv'))
        d2.to_csv(Path(self.config.root_dir, 'data_sources_homeoffice.csv'))
        d3.to_csv(Path(self.config.root_dir, 'data_sources_solanatest.csv'))

        logger.info(f"Data preprocessing completed and saved to {self.config.root_dir}")
        end_time = time.time()
        logger.info(f"Data preprocessing completed in {end_time - start_time} seconds") 