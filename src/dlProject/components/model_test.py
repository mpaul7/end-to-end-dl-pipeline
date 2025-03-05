import numpy as np
from pathlib import Path
from datetime import datetime

from dlProject.entity.config_entity import TestModelDlConfig
from dlProject.utils.features import *

import tensorflow as tf

from sklearn.metrics import confusion_matrix


class TestModelDl:
    def __init__(self, config: TestModelDlConfig, test_dataset: tf.data.Dataset):
        self.config = config
        self.params = config.params
        self.test_dataset = test_dataset
        self.params.confusion_matrix_file_name = Path(self.config.root_dir, self.config.model_file_name.split(".")[0] + "_confusion_matrix.csv" )

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
        import mlflow
        
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = "_".join(model_type for model_type in self.params.project.model_type)
        self.params.experiment_name = f"{model_name}_{len(stat_features_tr)}"
        
        with mlflow.start_run(run_name=f"{self.params.project.run_name}"):           
            _confusion_matrix = confusion_matrix(y_test, predictions)
            # for i in range(len(_confusion_matrix)):
            #     for j in range(len(_confusion_matrix)):
            #         mlflow.log_metric(f"confusion_matrix_{i}_{j}", _confusion_matrix[i][j])
            # mlflow.log_artifact(_confusion_matrix)
            # confusion_matrix_file_name = Path(test_model_dl_config.root_dir, test_model_dl_config.model_file_name.split(".")[0] + "_confusion_matrix.csv" )
            # _confusion_matrix.to_csv(self.params.confusion_matrix_file_name)
            # mlflow.log_artifact(self.params.confusion_matrix_file_name)
        return _confusion_matrix