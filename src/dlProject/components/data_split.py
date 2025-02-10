import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split                
from dlProject.entity.config_entity import DataSplitConfig
from dlProject.utils.common import read_file
from dlProject import logger

class DataSplit:
    def __init__(self, config: DataSplitConfig):
        self.config = config
        
    def split_data(self):
        file = Path(self.config.data_source_dir, self.config.data_file_name)
        df = read_file(file)
        train_df, test_df = train_test_split(df, test_size=self.config.test_size, random_state=42, stratify=df[self.config.label_column])
        train_file = Path(self.config.root_dir, f"{self.config.data_file_name.split('.')[0]}_train.csv")
        test_file = Path(self.config.root_dir, f"{self.config.data_file_name.split('.')[0]}_test.csv")
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Test data shape: {test_df.shape}")