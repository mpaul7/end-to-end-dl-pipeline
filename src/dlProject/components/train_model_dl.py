import warnings
warnings.filterwarnings('ignore')

import os
import click
import json
from pathlib import Path
from datetime import datetime
from dlProject.components import utils_dl
from dlProject.components.dl_models import DLModels
from dlProject.entity.config_entity import TrainModelDlConfig

import tensorflow as tf
# from tensorflow.keras.models import model_from_json
class TrainModelDl:
    
    def __init__(self, config: TrainModelDlConfig):
        self.config = config
        self.params = config.params
        
    def train_model_dl(self):
        """ Train DL model
            creates model architecture, trains the model and saves the model
            Args:
            train_data_file (str): Path to the training data file
            config_file (str): Path to the configuration file

    """
                
        current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name = "_".join(model_type for model_type in self.params.project.model_type)
        self.params.model_name = model_name
    
        self.params.model_plot = Path(self.config.root_dir, f"{model_name}_{current_datetime}.png")
        self.params.model_h5_path = Path(self.config.root_dir, f"{model_name}_{self.params.model_params.epochs}_{self.params.model_params.learning_rate}_{current_datetime}.h5")
        print(self.params.model_h5_path)
        self.params.model_json_path = Path(self.config.root_dir, f"{model_name}_{self.params.model_params.epochs}_{self.params.model_params.learning_rate}_{current_datetime}.json")
        self.params.output_units = len(self.params.labels)

        
        self.params.experiment_name = f"cnn_mlp_stat_solana_data_nfs_ext_2024-02-08_v1"
        
        _model_arch = "/home/mpaul/projects/mpaul/mai/models/mlp_120_0.01_20250211223752.json"
        with open(_model_arch) as f:
            model_arch = json.load(f)
        model = tf.keras.models.model_from_json(model_arch)
        """ Model Summary """
        model.summary()
        
        """ Train DL model
            Return: trained model
        """
        
        train_file = Path(self.config.data_source_dir, self.config.train_data_file_name)
           
        _model = DLModels(
            train_file=train_file,
            params=self.params,
            model=model,
            test_file=None,
            trained_model_file=None
        )
        
        model = _model.train_model()
        # model.save(params['model_h5_path'], save_format='h5')