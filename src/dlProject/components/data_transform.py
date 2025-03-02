
import time
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from dlProject import logger
from dlProject.utils.common import read_file
from dlProject.utils.features import *
from dlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    # tranalyzer transformed data
    def transform_data(self):
        start_time = time.time()
        file = Path(self.config.data_source_dir, self.config.data_file_name)
        df = read_file(file)
        
        # filter out rows with sport or dport as 53 or 5353, and rows with refined_app_label as null
        df = df[~((df['sport'].isin([53, 5353]) | df['dport'].isin([53, 5353])) | df['refined_app_label'].isna())]
        logger.info(f"Dataset shape: {df.shape}")
        
        # Fill NaN values with 0 across all columns
        df = df.fillna(0)
        
        # Verify the changes
        # logger.info("After filling NaN values:")
        # logger.info("Total rows with NaN values:", df.isna().any(axis=1).sum())
        # logger.info("\nNaN values per column:")
        # logger.info(df.isna().sum()[df.isna().sum() > 0])
        
        logger.info(f"Transforming data using StandardScaler")
        scaler = StandardScaler()
        df[stat_features_tr] = scaler.fit_transform(df[stat_features_tr])
        df['stat_features_tr_cnn'] = df[stat_features_tr].values.tolist()
        file_name = Path(self.config.root_dir, (self.config.data_file_name).split(".")[0] + "_transformed.csv")
        df.to_csv(file_name, index=False)
        logger.info(f"Data transformation completed and saved to {file_name}")
        end_time = time.time()
        logger.info(f"Data transformation completed in {end_time - start_time} seconds")
    
    # general transformed data
    # def transform_data(self):
        
    #     file = Path(self.config.data_source_dir, self.config.data_file_name)
    #     df = read_file(file)
        
    #     # Get dataset summary
    #     logger.info(f"Dataset shape: {df.shape}")
    #     initial_rows = len(df)
    #     df = df.dropna(subset=[self.config.label_column])
    #     final_rows = len(df)
    #     dropped_rows = initial_rows - final_rows
    #     if dropped_rows > 0:
    #         logger.info(f"Dropped {dropped_rows} rows containing null values. Rows reduced from {initial_rows} to {final_rows}")
    #     else:
    #         logger.info("No rows were dropped - no null values found in the dataset")
    #     scaler = StandardScaler()
    #     df[stat_features_twc] = scaler.fit_transform(df[stat_features_twc])
    #     df['stat_features'] = df[stat_features_twc].values.tolist()
    #     file_name = Path(self.config.root_dir, (self.config.data_file_name).split(".")[0] + "_transformed.csv")
    #     df.to_csv(file_name, index=False)
    #     logger.info(f"Data transformation completed and saved to {file_name}")