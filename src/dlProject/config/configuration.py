from pathlib import Path
from dlProject.constants import *
from dlProject.utils.common import read_yaml, create_directories
from dlProject.utils.features import *
from dlProject.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataSplitConfig, TrainModelDlConfig

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = Path(CONFIG_FILE_PATH),
            params_filepath = Path(PARAMS_FILE_PATH),
            schema_filepath = Path(SCHEMA_FILE_PATH)
        ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        self.params.features.stat_features = stat_features_twc
        self.params.labels.target_labels = target_labels
        self.params.labels.target_column = target_column
        self.params.features.seq_packet_feature = seq_packet_feature
        
        create_directories([self.config.artifacts_root])
        

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config_data_ingestion = self.config.data_ingestion
        create_directories([config_data_ingestion.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config_data_ingestion.root_dir,
            data_source_dir=config_data_ingestion.data_source_dir,
            data_file_name=config_data_ingestion.data_file_name
        )
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config_data_transformation = self.config.data_transformation
        create_directories([config_data_transformation.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config_data_transformation.root_dir,
            data_source_dir=config_data_transformation.data_source_dir,
            data_file_name=config_data_transformation.data_file_name,
            label_column=config_data_transformation.label_column
        )
        return data_transformation_config

    def get_data_split_config(self) -> DataSplitConfig:
        config_data_split = self.config.data_split
        create_directories([config_data_split.root_dir])
        data_split_config = DataSplitConfig(
            root_dir=config_data_split.root_dir,
            data_source_dir=config_data_split.data_source_dir,
            data_file_name=config_data_split.data_file_name,
            label_column=config_data_split.label_column,
            test_size=config_data_split.test_size
        )
        return data_split_config
    
    def get_train_model_dl_config(self) -> TrainModelDlConfig:
        config_train_model_dl = self.config.model_trainer
        create_directories([config_train_model_dl.root_dir])
        train_model_dl_config = TrainModelDlConfig(
            root_dir=config_train_model_dl.root_dir,
            data_source_dir=config_train_model_dl.data_source_dir,
            train_data_file_name=config_train_model_dl.train_data_file_name,
            test_data_file_name=config_train_model_dl.test_data_file_name,
            model_name=config_train_model_dl.model_name,
            params=self.params
        )
        return train_model_dl_config