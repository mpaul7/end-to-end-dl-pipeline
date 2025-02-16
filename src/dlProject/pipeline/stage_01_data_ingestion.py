import json
from pathlib import Path
from dlProject import logger
from datetime import datetime

from dlProject.commons.create_tf_dataset import create_train_test_dataset_tf
from dlProject.utils.classification_report import getClassificationReport
from dlProject.config.configuration import ConfigurationManager
from dlProject.components.data_ingestion import DataIngestion
from dlProject.components.data_transformation import DataTransformation
from dlProject.components.data_split import DataSplit
from dlProject.components.build_model import BuildModel
from dlProject.components.train_model_dl import TrainModelDl
from dlProject.components.test_model_dl import TestModelDl

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transform_data()
        
class DataSplitPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_split_config = config_manager.get_data_split_config()
        data_split = DataSplit(config=data_split_config)
        data_split.split_data()
        
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
            
       
class TrainModelDlPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        train_model_dl_config = config_manager.get_train_model_dl_config()
        train_dataset, val_dataset = create_train_test_dataset_tf(
            data_file=Path(train_model_dl_config.data_source_dir, train_model_dl_config.train_data_file_name),
            params=train_model_dl_config.params,
            train=True,
            evaluation=False
        )
        logger.info(f"\nTF Train dataset created")
        train_model_dl = TrainModelDl(config=train_model_dl_config, train_dataset=train_dataset, val_dataset=val_dataset)
        model = train_model_dl.train_model_dl()
        
        """ Save model """
        model.save(Path(train_model_dl_config.root_dir, f"{train_model_dl_config.train_model_file_name}"))
        logger.info(f"\nModel saved to {Path(train_model_dl_config.root_dir, f"{train_model_dl_config.train_model_file_name}")}")
        
class TestModelDlPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        test_model_dl_config = config_manager.get_test_model_dl_config()
        """Create test dataset"""
        test_dataset = create_train_test_dataset_tf(
            data_file=Path(test_model_dl_config.data_source_dir, test_model_dl_config.test_data_file_name),
            params=test_model_dl_config.params,
            train=False,
            evaluation=True
        )
        test_model_dl = TestModelDl(config=test_model_dl_config, test_dataset=test_dataset)
        confusion_matrix = test_model_dl.test_model_dl()
        """Classification Report"""
        classification_report = getClassificationReport(
            _confusion_matrix=confusion_matrix,
            traffic_classes=test_model_dl_config.params.labels.target_labels
        )
        confusion_matrix_file_name = Path(test_model_dl_config.root_dir, test_model_dl_config.model_file_name.split(".")[0] + "_confusion_matrix.csv" )
        print("\n", classification_report)
        classification_report.to_csv(confusion_matrix_file_name)
        logger.info(f"\nClassification report saved to {confusion_matrix_file_name}")