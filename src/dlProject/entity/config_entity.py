from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    root_dir: str
    data_source_dir: str
    data_file_name: str
    
@dataclass
class DataPreprocessingConfig:
    root_dir: str
    data_source_dir: str
    data_file_name: str
    
@dataclass
class DataTransformationConfig:
    root_dir: str
    data_source_dir: str
    data_file_name: str
    label_column: str
    
@dataclass
class DataSplitConfig:
    root_dir: str
    data_source_dir: str
    data_file_name: str
    params: dict
    
@dataclass
class FeatureSelectionConfig:
    root_dir: str
    data_source_dir: str
    train_data_file_name: str
    test_data_file_name: str
    params: dict
    
@dataclass
class BuildModelConfig:
    root_dir: str
    model_json: str
    model_plot: str
    params: dict
@dataclass
class TrainModelDlConfig:
    root_dir: str
    data_source_dir: str
    train_data_file_name: str
    model_source_dir: str
    model_file_name: str
    train_model_file_name: str
    params: dict
    
@dataclass
class TestModelDlConfig:
    root_dir: str
    data_source_dir: str
    # test_data_file_name: str
    cross_data_source_dir: str
    test_files: list
    model_dir: str
    model_file_name: str
    params: dict