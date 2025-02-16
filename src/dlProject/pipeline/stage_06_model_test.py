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