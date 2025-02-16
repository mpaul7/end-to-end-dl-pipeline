import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np

from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from keras.callbacks import ReduceLROnPlateau, Callback
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


import mlflow
from dlProject.components import utils_dl
# import utils_dl
import platform

class ModelBuilder:
    def __init__(self, params: dict):
        """Initialize ModelBuilder with configuration parameters.
        
        Args:
            params (dict): Model configuration parameters
        """
        self.params = params

    def build_model(self) -> Model:
        """Builds and returns the complete model architecture.
        
        Returns:
            Model: Compiled Keras model ready for training
        """
        input_branches = self._build_model_branches()
        model_arch = self._create_final_model(input_branches)
        """ Visualize model """
        tf.keras.utils.plot_model(model_arch, self.params['model_plot'], show_shapes=True)
        return model_arch   

    def _build_model_branches(self) -> dict:
        """Builds the input branches based on model types specified in params.
        
        Returns:
            dict: Dictionary containing model inputs and layers
        """
        input_branches = {'inputs': [], 'layer': []}
        model_types = self.params['model_types']
        if len(model_types) > 0:
            for model_type in model_types:
                model_type = model_type.strip()
                if model_type == 'lstm':
                    inputs, layers = utils_dl.create_dl_model_lstm(self.params)
                elif model_type == 'mlp':
                    inputs, layers = utils_dl.create_dl_model_mlp(self.params)
                elif model_type == 'cnn':
                    inputs, layers = utils_dl.create_dl_model_cnn(self.params)
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
            x = layers.Dense(units=self.params["dense"]["dense_units_list"][i], 
                            kernel_initializer=utils_dl.KERAS_INITIALIZER[self.params["initializer"]],
                            name=f'final_dense_{i}'
                            )(x)
            x = layers.LeakyReLU(name=f'dense_leaky_relu_{i}')(x)
            x = layers.Dropout(self.params['dropout_rate'], name=f'dense_dropout_{i}')(x)
        
        # Add final dense layers
        for i in range(self.params["final_dense"]["num_final_dense"]):
            x = layers.Dense(units=self.params["final_dense"]["final_dense_units_list"][i], 
                            kernel_initializer=utils_dl.KERAS_INITIALIZER[self.params["initializer"]],
                            name=f'final_dense_{i}'
                            )(x)
            x = layers.LeakyReLU(name=f'final_dense_leaky_relu_{i}')(x)
            x = layers.Dropout(self.params['dropout_rate'], name=f'final_dense_dropout_{i}')(x)
        
        # Add output layer
        outputs = tf.keras.layers.Dense(self.params['output_units'], activation='softmax', name='softmax')(x)
        
        # Create final model
        model = models.Model(inputs=input_branches['inputs'], outputs=outputs)
        
        return model

class ModelBuilderTransferLearning_v1:
    def __init__(self, params: dict, output_units: int):
        """Initialize ModelBuilderTransferLearning with configuration parameters.
        
        Args:
            params (dict): Model configuration parameters
            output_units (int): Number of output units for the model
        """
        self.params = params
        self.output_units = output_units

    def build_model(self) -> Model:
        """Builds and returns the complete model architecture using transfer learning.
        
        Returns:
            Model: Compiled Keras model ready for training
        """
        return self._create_final_model()

    def _create_final_model(self) -> Model:
        """Creates the final model by adding dense layers on top of a pre-trained model.
        
        Returns:
            Model: Final Keras model
        """
        # Load the pre-trained model
        base_model = load_model('/home/mpaul/projects/mpaul/mai/models/mlp_lstm_cnn_20241203190916.h5')
        # base_model.trainable = False  # Freeze the base model layers
        for layer in base_model.layers:
            # print(layer.name)
            layer.trainable = False

        # Create a new model on top of the base model
        x = base_model.output
        print(x.name)

        # Add three dense layers with unique names
        # for i in range(3):
        # x = tf.keras.layers.Dense(units=self.params["encoder_dense_units_list"][0])(x)
        x = tf.keras.layers.Dense(units=128, activation='relu', name='new_dense_1')(x)
        # x = tf.keras.layers.LeakyReLU()(x)

        # Add output layer
        outputs = tf.keras.layers.Dense(7, activation='softmax', name='new_softmax_output')(x)

        # Create the final model
        model = models.Model(inputs=base_model.input, outputs=outputs)
        
        return model

class ModelBuilderTransferLearning:
    def __init__(self, params: dict, output_units: int):
        """Initialize ModelBuilderTransferLearning with configuration parameters.
        
        Args:
            params (dict): Model configuration parameters
            output_units (int): Number of output units for the model
        """
        self.params = params
        self.output_units = output_units

    def build_model(self) -> Model:
        """Builds and returns the complete model architecture using transfer learning.
        
        Returns:
            Model: Compiled Keras model ready for training
        """
        return self._create_final_model()

    def _create_final_model(self) -> Model:
        """Creates the final model by adding dense layers on top of a pre-trained model.
        
        Returns:
            Model: Final Keras model
        """
        print('Creates the final model by adding dense layers on top of a pre-trained model.')
        # Load the pre-trained model
        # base_model = load_model('/home/mpaul/projects/mpaul/mai/models/mlp_lstm_cnn_20241203190916.h5')
        
        # model # 8
        # base_a = load_model('/home/mpaul/projects/mpaul/mai/models/models_jan13/1_mlp_120_0.0001_20250113155400.h5')
        # base_b = load_model('/home/mpaul/projects/mpaul/mai/models/models_jan13/2_lstm_120_0.0001_20250113155937.h5')

        # model # 9
        # base_a = load_model('/home/mpaul/projects/mpaul/mai/models/models_jan13/1_mlp_120_0.0001_20250113155400.h5')
        # base_b = load_model('/home/mpaul/projects/mpaul/mai/models/models_jan10/3_cnn_100_seq_7e-05_20250112220441.h5')
        
        # model # 10
        base_a = load_model('/home/mpaul/projects/mpaul/mai/models/models_jan13/2c_lstm_300_0.001_20250114202244.h5')
        base_b = load_model('/home/mpaul/projects/mpaul/mai/models/models_jan13/4_cnn_120_0.0001_20250113204311.h5')
        
        print("Model A Layers:")
        for i, layer in enumerate(base_a.layers):
            print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")

        print("\nModel B Layers:")
        for i, layer in enumerate(base_b.layers):
            print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")

        # print("\nModel C Layers:")
        # for i, layer in enumerate(base_cnn.layers):
        #     print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")

        # print(base_mlp.layers[64].output)   
        
        # model 8 - mlp - 65 , lstm - 13
        # model 9 - mlp - 65, cnn 26
        # model 10 - lstm - 13, cnn - 19
        

        partial_model_a = Model(inputs=base_a.input, outputs=base_a.layers[11].output)  
        partial_model_b = Model(inputs=base_b.input, outputs=base_b.layers[24].output) 
        # base_model.trainable = False  # Freeze the base model layers
        for layer in partial_model_a.layers:
            if not isinstance(layer, layers.InputLayer):
                layer._name = layer.name + '_a'
            layer.trainable = False
        
        for layer in partial_model_b.layers:
            if not isinstance(layer, layers.InputLayer):
                layer._name = layer.name + '_b'
            layer.trainable = False

        # for layer in partial_model_c.layers:
        #     if not isinstance(layer, layers.InputLayer):
        #         layer._name = layer.name + '_c'
        #     layer.trainable = False

        # Define the input shape for both models
        input_A = partial_model_a.input
        input_B = partial_model_b.input
        # input_C = partial_model_c.input

        # Extract features from the partial models
        features_A = partial_model_a.output
        features_B = partial_model_b.output
        # features_C = partial_model_c.output

        # Concatenate the features extracted from both models
        combined_features = layers.concatenate([features_A, features_B])

        # Classifier part
        # initializer = KERAS_INITIALIZER.get('he_uniform')()

        x = layers.Dropout(self.params["encoder_dense_dropout_rate"])(combined_features)

        for i in range(self.params.get("num_encoder_dense", 1)):
            x = layers.Dense(units=self.params.get("encoder_dense_units_list", [128])[i])(x)
            # x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)

        for i in range(self.params.get("num_final_dense", 1)):
            x = layers.Dense(units=self.params.get("final_dense_units_list", [128])[i], 
            # kernel_initializer=initializer
            )(x)
            # x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(self.params.get("final_dense_dropout_rate", 0.1))(x)

        outputs = layers.Dense(self.output_units, activation='softmax', name='softmax')(x)

        inputs = [input_A, input_B]    
        outputs = outputs
        combined_model = Model(inputs=inputs, outputs=outputs)
        
        return combined_model

class LearningRateRecorder(Callback):
    
    def on_train_begin(self, logs={}):
        self.lrate_list = list()
    
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lrate = float(tf.keras.backend.get_value(optimizer.lr))
        self.lrate_list.append(lrate)

class DLModels:
    
    def __init__(self, train_file: str, params: dict, model: Model, test_file: str, trained_model_file: str):
        """Initialize DLModels with configuration parameters.
        
        Args:
            model_type (str): Type of model to train ('lstm', 'mlp', 'cnn', or combination)
            train_file (str): Path to the training data file
            config (str): Path to the model configuration JSON file
            test_file (str): Path to the test data file for validation
        """
        self.train_file = train_file
        self.model = model 
        self.params = params
        self.test_file = test_file
        self.trained_model_file = trained_model_file





    def test_model(self):
        """Model Evaluation
        
        outputs a confusion matrix
        """
        
        """Load Trained Model"""
        loaded_model = load_model(self.trained_model_file)
        
        """Create test dataset"""
        test_dataset = utils_dl.create_train_test_dataset_tf(
            data_file=self.test_file,
            params=self.params,
            train=False,
            evaluation=True
        )
        test_dataset = test_dataset.batch(128)
        
        """predict test data using loaded model"""
        y_test = np.concatenate([y for _, y in test_dataset], axis=0).argmax(axis=1)
        predictions = loaded_model.predict(test_dataset)
        predictions = predictions.argmax(axis=1)
        
        """Confusion Matrix"""
        labels = self.params['labels']
        labels = [labels[i] for i in np.unique(predictions).tolist()]
        _confusion_matrix_flow_count = confusion_matrix(y_test, predictions)
        
        
        print(_confusion_matrix_flow_count)
        matrix = utils_dl.getClassificationReport(
            _confusion_matrix=_confusion_matrix_flow_count,
            traffic_classes=labels
        )
        
        return matrix
