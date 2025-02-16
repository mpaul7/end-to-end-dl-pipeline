import numpy as np
from pathlib import Path
from datetime import datetime

from dlProject.entity.config_entity import TestModelDlConfig

import tensorflow as tf
from sklearn.metrics import confusion_matrix


class TestModelDl:
    def __init__(self, config: TestModelDlConfig, test_dataset: tf.data.Dataset):
        self.config = config
        self.params = config.params
        self.test_dataset = test_dataset


    def test_model_dl(self):
        """Model Evaluation
        
        outputs a confusion matrix
        """
        
        """Load Trained Model"""
        model_file_name = Path(self.config.model_dir, self.config.model_file_name)
        loaded_model = tf.keras.models.load_model(model_file_name)
        

        test_dataset = self.test_dataset.batch(128)
        
        """predict test data using loaded model"""
        y_test = np.concatenate([y for _, y in test_dataset], axis=0).argmax(axis=1)
        predictions = loaded_model.predict(test_dataset)
        predictions = predictions.argmax(axis=1)
        
        """Confusion Matrix"""
        _confusion_matrix = confusion_matrix(y_test, predictions)

        return _confusion_matrix