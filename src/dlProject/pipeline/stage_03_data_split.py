from dlProject import logger

from dlProject.config.configuration import ConfigurationManager
from dlProject.components.data_split import DataSplit

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