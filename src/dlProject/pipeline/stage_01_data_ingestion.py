from pathlib import Path
from dlProject.config.configuration import ConfigurationManager
from dlProject.components.data_ingestion import DataIngestion
from dlProject.components.data_transformation import DataTransformation
from dlProject.components.data_split import DataSplit
from dlProject.components.train_model_dl import TrainModelDl
from dlProject.components.test_model_dl import TestModelDl
from dlProject import logger
from dlProject.components import utils_dl
from datetime import datetime
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
        
class TrainModelDlPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        train_model_dl_config = config_manager.get_train_model_dl_config()
        train_dataset, val_dataset = utils_dl.create_train_test_dataset_tf(
            data_file=Path(train_model_dl_config.data_source_dir, train_model_dl_config.train_data_file_name),
            params=train_model_dl_config.params,
            train=True,
            evaluation=False
        )
        train_model_dl = TrainModelDl(config=train_model_dl_config, train_dataset=train_dataset, val_dataset=val_dataset)
        model = train_model_dl.train_model_dl()
        
        """ Save model """
        current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = "_".join(model_type for model_type in train_model_dl_config.params.project.model_type)
        model_h5_path = Path(train_model_dl_config.root_dir, f"{model_name}_{train_model_dl_config.params.model_params.epochs}_{train_model_dl_config.params.model_params.learning_rate}_{current_datetime}.h5")
        model.save(model_h5_path)
        
class TestModelDlPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        test_model_dl_config = config_manager.get_test_model_dl_config()
        """Create test dataset"""
        test_dataset = utils_dl.create_train_test_dataset_tf(
            data_file=Path(test_model_dl_config.data_source_dir, test_model_dl_config.test_data_file_name),
            params=test_model_dl_config.params,
            train=False,
            evaluation=True
        )
        test_model_dl = TestModelDl(config=test_model_dl_config, test_dataset=test_dataset)
        confusion_matrix = test_model_dl.test_model_dl()
        """Classification Report"""
        classification_report = utils_dl.getClassificationReport(
            _confusion_matrix=confusion_matrix,
            traffic_classes=test_model_dl_config.params.labels.target_labels
        )
        confusion_matrix_file_name = Path(test_model_dl_config.root_dir, test_model_dl_config.model_file_name.split(".")[0] + "_confusion_matrix.csv" )
        print(classification_report)
        classification_report.to_csv(confusion_matrix_file_name)