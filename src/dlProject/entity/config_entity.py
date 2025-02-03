from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    root_dir: str
    data_source_dir: str
    data_file_name: str