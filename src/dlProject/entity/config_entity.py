from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
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
    label_column: str
    test_size: float