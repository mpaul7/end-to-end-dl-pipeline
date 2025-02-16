import os

import zipfile as zip
from pathlib import Path
import urllib.request as request

from dlProject import logger
from dlProject.utils.common import get_size 
from dlProject.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_file(self):

        file_path = Path(self.config.data_source_dir, self.config.data_file_name)
        destination_path = Path(self.config.root_dir)

        os.system(f"cp {file_path} {destination_path}")
       
        logger.info(f"{file_path} downloaded with the following info: {file_path}")