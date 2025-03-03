from pathlib import Path
import time
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier  
from dlProject.utils.features import *
from dlProject.entity.config_entity import FeatureSelectionConfig
from dlProject.utils.common import read_file
from dlProject import logger

class FeatureSelection:
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.params = config.params
        self.train_data_file_name = config.train_data_file_name
        self.test_data_file_name = config.test_data_file_name
        
    def feature_selection(self):
        start_time = time.time()
        train_file = Path(self.config.data_source_dir, self.config.train_data_file_name)
        test_file = Path(self.config.data_source_dir, self.config.test_data_file_name)
        train_df = read_file(train_file)
        test_df = read_file(test_file)
        
        X_train = train_df[stat_features_tr]
        y_train = train_df[target_column]
        
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        number_of_features_to_select = 50
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
        logger.info(f"Feature selection completed in {time.time() - start_time} seconds")
     
        
        
        

        
        # train_df, test_df = train_test_split(df, test_size=self.params.data_split.test_size, random_state=42, #stratify=df[self.config.label_column]
        # feature_importances_df = pd.DataFrame({'feature': train_df.drop(columns=[self.config.label_column]).columns, 'importance': feature_importances})
        # feature_importances_df cd = feature_importances_df.sort_values(by='importance', ascending=False)
        # selected_features = feature_importances_df.head(number_of_features_to_select)['feature'].tolist()
        # train_df = train_df[selected_features + [self.config.label_column]]
        # test_df = test_df[selected_features + [self.config.label_column]]
        # train_df, test_df = train_test_split(df, test_size=self.params.data_split.test_size, random_state=42, #stratify=df[self.config.label_column]
        # )

        # train_file = Path(self.config.root_dir, f"{self.config.data_file_name.split('.')[0]}_train.csv")
        # test_file = Path(self.config.root_dir, f"{self.config.data_file_name.split('.')[0]}_test.csv")
        # train_df.to_csv(train_file, index=False)
        # test_df.to_csv(test_file, index=False)
        # logger.info(f"Train data shape: {train_df.shape}")
        # logger.info(f"Test data shape: {test_df.shape}")