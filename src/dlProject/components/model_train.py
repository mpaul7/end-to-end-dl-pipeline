import warnings
warnings.filterwarnings('ignore')

import json
import mlflow
import platform
import os
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy

from dlProject.utils.common import *
from dlProject.utils.features import *
from dlProject.entity.config_entity import TrainModelDlConfig

loss_function = {
    'categorical_crossentropy': CategoricalCrossentropy(),
    'sparse_categorical_crossentropy': SparseCategoricalCrossentropy()
}

class TrainModelDl:
    
    def __init__(self, config: TrainModelDlConfig, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset):
        self.config = config
        self.params = config.params
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        model_name = "_".join(model_type for model_type in self.params.project.model_type)
        self.params.model_name = model_name
        self.params.model_plot = Path(self.config.root_dir, f"{model_name}.png")
        self.params.output_units = len(self.params.labels)
        
        self.params.experiment_name = f"{model_name}_{len(stat_features_tr)}"
        self.params.model_h5_path = Path(self.config.root_dir, self.config.train_model_file_name)
        

    def train_model_dl(self):
        """ Train DL model
            creates model architecture, trains the model and saves the model
            Args:
            train_data_file (str): Path to the training data file
            config_file (str): Path to the configuration file

    """
        _model_arch = Path(self.config.model_source_dir, self.config.model_file_name)
        
        with open(_model_arch) as f:
            model_arch = json.load(f)
        model = tf.keras.models.model_from_json(model_arch)
        
        
        """ Model Summary """
        model.summary()
        
        """ Visualize model """
        tf.keras.utils.plot_model(model, self.params.model_plot, show_shapes=True)
        
        """ Compile model """
        model.compile(
            loss=loss_function[self.params.model_params.loss_function],
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params.model_params.learning_rate),
            metrics=self.params.model_params.metrics
        )
        
        """ Apply batching """
        train_dataset = self.train_dataset.batch(self.params.model_params.train_batch_size)
        val_dataset = self.val_dataset.batch(self.params.model_params.test_batch_size)
        
        """ Setup callbacks """
        callbacks = self._setup_callbacks()
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with mlflow.start_run(run_name=f"{self.params.project.run_name}"):
            # Log model parameters
            mlflow.log_params({
                'learning_rate': self.params.model_params.learning_rate,
                'batch_size': self.params.model_params.train_batch_size,
                'epochs': self.params.model_params.epochs,
                'steps_per_epoch': self.params.model_params.steps_per_epoch,
                'dropout_rate': self.params.model_params.dropout_rate,
                'regularizer': self.params.model_params.regularizer,
            })
            
            # Log system metrics
            mlflow.log_param("system_info", {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
            })

            """ Train model """
            history = model.fit(
                train_dataset,
                epochs=self.params.model_params.epochs,
                steps_per_epoch=self.params.model_params.steps_per_epoch,
                validation_data=val_dataset,
                callbacks=callbacks
            )
            
            for epoch in range(len(history.history['accuracy'])):
                mlflow.log_metrics({
                    'training_accuracy': history.history['accuracy'][epoch],
                    'training_loss': history.history['loss'][epoch],
                    'validation_accuracy': history.history['val_accuracy'][epoch], 
                    'validation_loss': history.history['val_loss'][epoch]
                }, step=epoch)
                
            """ Save model """
            model.save(self.params.model_h5_path)
            logger.info(f"\nModel saved to {self.params.model_h5_path}")
            
            mlflow.tensorflow.log_model(model, "model", registered_model_name=self.params.model_name)
            mlflow.log_artifact(self.params.model_h5_path)
            
            return model

    def _setup_callbacks(self) -> list:
        """Sets up training callbacks based on configuration.
        
        Args:
            params (dict): Model configuration parameters
            
        Returns:
            list: List of Keras callbacks
        """
        callbacks = []
        
        # Add TensorBoard callback if log directory is specified
        log_dir = f"{self.params.project.project_home}/{self.params.project.log_dir}"
        if log_dir:
            callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
        
        """ Add early stopping if enabled """
        if self.params.model_params.is_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor=self.params.model_params.early_stopping.monitor, 
                patience=self.params.model_params.early_stopping.patience,
                )
            )
        
        return callbacks