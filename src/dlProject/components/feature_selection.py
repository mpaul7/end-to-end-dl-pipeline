from pathlib import Path
import time
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier  
from dlProject.utils.features import *
from dlProject.entity.config_entity import FeatureSelectionConfig
from dlProject.utils.common import read_file
from dlProject import logger
from sklearn.preprocessing import LabelEncoder
import pickle

class FeatureSelection:
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.params = config.params
        self.train_data_file_name = config.train_data_file_name
        self.test_data_file_name = config.test_data_file_name
        
    def feature_selection_v1(self):
        # Get feature lists from configuration
        target_column = self.params.labels.target_column
        # stat_features_tr = stat_features_train
        
        logger.info(f"Starting feature selection process...")
        logger.info(f"Total number of initial features: {len(stat_features_tr)}")
        logger.info(f"Target column: {target_column}")
        start_time = time.time()
        train_file = Path(self.config.data_source_dir, self.config.train_data_file_name)
        test_file = Path(self.config.data_source_dir, self.config.test_data_file_name)
        train_df = read_file(train_file)
        logger.info(f"Train data shape: {train_df.shape}")
        test_df = read_file(test_file)
        
        X_train = train_df[stat_features_tr]
        logger.info(f"X_train shape: {X_train.shape}")
        y_train = train_df[target_column]
        
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        number_of_features_to_select = int(0.7 * X_train.shape[1])
        logger.info(f"Number of features to select: {number_of_features_to_select}")
        rfe = RFE(estimator, n_features_to_select=number_of_features_to_select)
        X_selected = rfe.fit_transform(X_train, y_train)
        
        # Get the selected feature names
        selected_features = X_train.columns[rfe.support_].tolist()
        logger.info(f"Selected features: {selected_features}")
        
        # Save selected features to a file in the root directory
        selected_features_file = Path(self.config.root_dir, "selected_features.txt")
        with open(selected_features_file, 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
        logger.info(f"Selected features saved to: {selected_features_file}")
        logger.info(f'total time taken: {time.time() - start_time}')
        return X_selected, selected_features
    
    """Description:
    This function performs feature selection using Random Forest and XGBoost.
    It supports GPU acceleration for faster computation.
    XGBoost that supports GPU is : pip install xgboost==1.7.3
    Set the estimator hyperparameters as shown below.
    It selects features based on their importance scores and saves the selected features to a file.
    """
    def feature_selection(self):
        logger.info("Starting feature selection process...")
        
        logger.info(f"Starting feature selection process...")
        logger.info(f"Total number of initial features: {len(stat_features_tr)}")
        logger.info(f"Target column: {target_column}")
        start_time = time.time()
        train_file = Path(self.config.data_source_dir, self.config.train_data_file_name)
        test_file = Path(self.config.data_source_dir, self.config.test_data_file_name)
        train_df = read_file(train_file)
        logger.info(f"Train data shape: {train_df.shape}")
        test_df = read_file(test_file)
        
        X_train = train_df[stat_features_tr]
        logger.info(f"X_train shape: {X_train.shape}")
        y_train = train_df[target_column]
        
        logger.info(f"Total number of initial features: {len(stat_features_tr)}")
        logger.info(f"Target column: refined_app_label")
        logger.info(f"Train data shape: {train_df.shape}")
        
        # Convert string labels to numbers using LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        # Save the label encoder for later use
        encoder_path = Path(self.config.root_dir, "label_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(le, f)
        
        logger.info(f"X_train shape: {X_train.shape}")
        number_of_features_to_select = int(0.7 * X_train.shape[1])
        logger.info(f"Number of features to select: {number_of_features_to_select}")
        
        # Create and fit RFE with XGBoost
        estimator = xgb.XGBClassifier(
            n_estimators=100,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            verbosity=1,
            gpu_id=0,
            eval_metric='logloss',
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=number_of_features_to_select,
            step=1
        )
        
        # Fit and transform the data
        X_selected = rfe.fit_transform(X_train, y_train_encoded)
        
        # Get selected feature names
        selected_features = np.array(stat_features_tr)[rfe.support_]
        
        # Save selected features
        selected_features_path = Path(self.config.root_dir, "selected_features.pkl")
        with open(selected_features_path, 'wb') as f:
            pickle.dump(selected_features, f)
        
        logger.info(f"Selected {len(selected_features)} features")
        logger.info("Feature selection completed successfully")
        logger.info(f"Selected features: {selected_features}")
        logger.info(f'total time taken: {time.time() - start_time}')
        return X_selected, selected_features