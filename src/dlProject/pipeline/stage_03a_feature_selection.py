import json
from pathlib import Path
from dlProject import logger
from datetime import datetime

from dlProject.config.configuration import ConfigurationManager
from dlProject.components.feature_selection import FeatureSelection

class FeatureSelectionPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        feature_selection_config = config_manager.get_feature_selection_config()
        feature_selection = FeatureSelection(config=feature_selection_config)
        feature_selection.feature_selection()

if __name__ == '__main__':
    try:    
        logger.info(f">>>>>> stage Feature Selection started <<<<<<")
        obj = FeatureSelectionPipeline()
        obj.main()
        logger.info(f">>>>>> stage Feature Selection completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e