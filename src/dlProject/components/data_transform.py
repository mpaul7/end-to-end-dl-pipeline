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
        df = df[df['refined_app_label'].isin(target_labels)]
        df = df[stat_features_tr + ['refined_app_label', 'data_source'] ]
        # Check for columns with all zero values
        zero_columns = df[stat_features_tr].columns[(df[stat_features_tr] != 0).all()].tolist()
        if zero_columns:
            logger.info(f"Columns with all zero values: {zero_columns}")
        else:
            logger.info("No columns found with all zero values")


        # filter out rows with sport or dport as 53 or 5353, and rows with refined_app_label as null
        # df = df[~((df['sport'].isin([53, 5353]) | df['dport'].isin([53, 5353])) | df['refined_app_label'].isna())]

        logger.info(f"Dataset shape before dropping null values: {df.shape}")
        df.dropna(inplace=True)
        logger.info(f"Dataset shape after dropping null values: {df.shape}")

        logger.info(f"Transforming data using StandardScaler")
        scaler = StandardScaler()
        df[stat_features_tr] = scaler.fit_transform(df[stat_features_tr])
        df['stat_features'] = df[stat_features_tr].values.tolist()

        # df = df[['stat_features', 'refined_app_label', 'data_source']].copy()
        file_name = Path(self.config.root_dir, (self.config.data_file_name).split(".")[0] + "_transformed.parquet")
        df.to_parquet(file_name, index=False)
        logger.info(f"Data transformation completed and saved to {file_name}")
        end_time = time.time()
        logger.info(f"Data transformation completed in {end_time - start_time} seconds")
    