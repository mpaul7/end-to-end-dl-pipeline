import json
from pathlib import Path
from dlProject import logger
from datetime import datetime

from dlProject.commons.create_tf_dataset import create_train_test_dataset_tf
from dlProject.utils.classification_report import getClassificationReport
from dlProject.config.configuration import ConfigurationManager
from dlProject.components.data_ingest import DataIngestion
from dlProject.components.data_transform import DataTransformation
from dlProject.components.data_split import DataSplit
from dlProject.components.model_build import BuildModel
from dlProject.components.model_train import TrainModelDl
from dlProject.components.model_test import TestModelDl

class DataSplitPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_split_config = config_manager.get_data_split_config()
        data_split = DataSplit(config=data_split_config)
        data_split.split_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage Data Split started <<<<<<")
        obj = DataSplitPipeline()
        obj.main()
        logger.info(f">>>>>> stage Data Split completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e