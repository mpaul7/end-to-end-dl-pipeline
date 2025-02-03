from dlProject.config.configuration import ConfigurationManager
from dfProject.components.data_ingestion import DataIngestion
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

