
from dlProject import logger

from dlProject.config.configuration import ConfigurationManager
from dlProject.components.data_ingest import DataIngestion

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage Data Ingestion started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage Data Ingestion completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e