
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import tensorflow as tf
from keras import layers
from keras import models
from keras import Model

from dlProject.entity.config_entity import BuildModelConfig
from dlProject.commons.create_mlp_model import create_dl_model_mlp
from dlProject.commons.create_lstm_model import create_dl_model_lstm
from dlProject.commons.create_cnn_model import create_dl_model_cnn
from dlProject.utils.features import *
# from dlProject.components import utils_dl
KERAS_INITIALIZER = {
        'default': tf.keras.initializers.GlorotUniform,
        'he_uniform': tf.keras.initializers.HeUniform,
        'he_normal': tf.keras.initializers.HeNormal
}
class BuildModel:
    def __init__(self, config: BuildModelConfig):
        """Initialize ModelBuilder with configuration parameters.
        
        Args:
            params (dict): Model configuration parameters
        """
        self.config = config
        self.params = config.params
        self.params.output_units = len(self.params.labels.target_labels)
        # self.params['output_units'] = len(config.params.labels.target_labels)

    def build_model(self) -> Model:
        """Builds and returns the complete model architecture.
        
        Returns:
            Model: Compiled Keras model ready for training
        """
        input_branches = self._build_model_branches()
        model_arch = self._create_final_model(input_branches)
        
        """ Visualize model """
        model_plot = Path(self.config.root_dir, self.config.model_plot)
        tf.keras.utils.plot_model(model_arch, model_plot, show_shapes=True)
        return model_arch   

    def _build_model_branches(self) -> dict:
        """Builds the input branches based on model types specified in params.
        
        Returns:
            dict: Dictionary containing model inputs and layers
        """
        input_branches = {'inputs': [], 'layer': []}
        model_types = self.params.project.model_type
        if len(model_types) > 0:
            for model_type in model_types:
                model_type = model_type.strip()
                if model_type == 'lstm':
                    inputs, layers = create_dl_model_lstm(self.params)
                elif model_type == 'mlp':
                    inputs, layers = create_dl_model_mlp(self.params)
                elif model_type == 'cnn':
                    inputs, layers = create_dl_model_cnn(self.params)
                input_branches['inputs'].append(inputs)
                input_branches['layer'].append(layers)
        else:
            raise ValueError(f"Invalid model type: model types list empty - valid model types: [mlp, lstm, cnn]")
        
        return input_branches

    def _create_final_model(self, input_branches: dict) -> Model:
        """Creates the final model by combining branches and adding dense layers.
        
        Args:
            input_branches (dict): Dictionary containing model inputs and layers
            
        Returns:
            Model: Final Keras model
        """
        # Combine branches if multiple
        if len(input_branches['layer']) > 1:
            x = tf.keras.layers.concatenate(input_branches['layer'])
        else:
            x = input_branches['layer'][0]
        
        # Add dense layers
        for i in range(self.params["dense"]["num_dense"]): # 2
            x = layers.Dense(units=self.params.dense.dense_units_list[i], 
                            kernel_initializer=KERAS_INITIALIZER[self.params.model_params.initializer],
                            name=f'final_dense_{i}'
                            )(x)
            x = layers.LeakyReLU(name=f'dense_leaky_relu_{i}')(x)
            x = layers.Dropout(self.params.model_params.dropout_rate, name=f'dense_dropout_{i}')(x)
        
        # Add final dense layers
        for i in range(self.params["final_dense"]["num_final_dense"]):
            x = layers.Dense(units=self.params.final_dense.final_dense_units_list[i], 
                            kernel_initializer=KERAS_INITIALIZER[self.params.model_params.initializer],
                            name=f'final_dense_{i}'
                            )(x)
            x = layers.LeakyReLU(name=f'final_dense_leaky_relu_{i}')(x)
            x = layers.Dropout(self.params['dropout_rate'], name=f'final_dense_dropout_{i}')(x)
        
        # Add output layer
        outputs = tf.keras.layers.Dense(self.params.output_units, activation='softmax', name='softmax')(x)
        
        # Create final model
        model = models.Model(inputs=input_branches['inputs'], outputs=outputs)
        
        return model