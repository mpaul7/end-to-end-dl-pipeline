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

class ModelBuilderPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        build_model_config = config_manager.get_build_model_config()
        build_model = BuildModel(config=build_model_config)
        model_arch  = build_model.build_model()
        model_json = model_arch.to_json()
        with open(f"{build_model_config.root_dir}/{build_model_config.model_json}", "w") as json_file:
            json.dump(model_json, json_file)
            
            """ Model Summary """
            model_arch.summary()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage Model Build started <<<<<<")
        obj = ModelBuilderPipeline()
        obj.main()
        logger.info(f">>>>>> stage Model Build completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e