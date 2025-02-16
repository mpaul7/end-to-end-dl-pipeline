import os

import zipfile as zip
import pandas as pd
from pathlib import Path
import urllib.request as request

from sklearn.preprocessing import StandardScaler

from dlProject import logger
from dlProject.utils.common import read_file
from dlProject.utils.features import *
from dlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def transform_data(self):
        
        file = Path(self.config.data_source_dir, self.config.data_file_name)
        df = read_file(file)
        
        # Get dataset summary
        logger.info(f"Dataset shape: {df.shape}")
        initial_rows = len(df)
        df = df.dropna(subset=[self.config.label_column])
        final_rows = len(df)
        dropped_rows = initial_rows - final_rows
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows containing null values. Rows reduced from {initial_rows} to {final_rows}")
        else:
            logger.info("No rows were dropped - no null values found in the dataset")
        scaler = StandardScaler()
        df[stat_features_twc] = scaler.fit_transform(df[stat_features_twc])
        df['stat_features'] = df[stat_features_twc].values.tolist()
        file_name = Path(self.config.root_dir, (self.config.data_file_name).split(".")[0] + "_transformed.csv")
        df.to_csv(file_name, index=False)
        logger.info(f"Data transformation completed and saved to {file_name}")