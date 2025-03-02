from dlProject import logger
from dlProject.config.configuration import ConfigurationManager
from dlProject.components.data_preprocessing import DataPreprocessing



class DataPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_preprocessing_config = config_manager.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.preprocess_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage Data Preparation started <<<<<<")
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage Data Preparation completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e