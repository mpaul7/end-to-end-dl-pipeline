from dlProject.config.configuration import ConfigurationManager
from dlProject.components.data_ingestion import DataIngestion
from dlProject.components.data_transformation import DataTransformation
from dlProject.components.data_split import DataSplit
from dlProject.components.train_model_dl import TrainModelDl
from dlProject import logger

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
        train_model_dl = TrainModelDl(config=train_model_dl_config)
        train_model_dl.train_model_dl()