
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from dlProject.utils.features import *
from dlProject.utils.common import read_file


def create_train_test_dataset_tf(data_file=None, params=None, train=None, evaluation=None):
    _df = read_file(data_file)
    df = _df[_df[params['labels']['target_column']].isin(params['labels']['target_labels'])]
    
    model_type = params['project']['model_type']
    features = []
    if 'mlp' in model_type:
        features.extend(params['features']['stat_features'])
    if 'lstm' in model_type:
        features.extend(params['features']['seq_packet_feature'])
    if 'cnn' in model_type:
        features.extend(params['features']['cnn_stat_feature'])
        
    X = df[features]
    _y = df.loc[:, [params['labels']['target_column']]]
    y = pd.get_dummies(_y)
    
    """ Create tf dataset """  
    
    def create_dataset(X, y):
        feat_dict = {}
        X_pktseq = {}
        if 'mlp' in model_type:
            X_flow = {name: np.stack(value) for name, value in X.loc[:, params['features']['stat_features']].items()}
            feat_dict['flow_features'] = X_flow
        if 'lstm' in model_type:
            X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params['features']['seq_packet_feature']].items()}
            feat_dict['pktseq_features'] = X_pktseq
        if 'cnn' in model_type:       
            X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params['features']['cnn_stat_feature']].items()}
            feat_dict['pktstat_features'] = X_pktseq
        
        ds_X = tf.data.Dataset.from_tensor_slices(X_pktseq, name='X')
        # ds_X = tf.data.Dataset.from_tensor_slices(feat_dict, name='X')
        ds_y = tf.data.Dataset.from_tensor_slices(y)

        tf_dataset = tf.data.Dataset.zip((ds_X, ds_y))
        # return tf_dataset
        return ds_X, ds_y, tf_dataset

    # def _create_balanced_tf_dataset(X, y, params):
    #     # Creates a DataFrame with a single column containing the predicted class labels
    #     # by finding the index of the maximum value in each row of one-hot encoded y
    #     df = pd.DataFrame({params.labels.target_column: y.idxmax(axis=1)})
    #     partials = []
    #     for _, group in df.groupby(params.labels.target_column):

    #         partials.append(create_dataset(X.loc[group.index], y.loc[group.index]).repeat())
    #     return tf.data.Dataset.sample_from_datasets(partials)

    if train:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.data_split.validation_size)
        # training_dataset = _create_balanced_tf_dataset(X_train, y_train, params)
        X_train, y_train, tf_dataset = create_dataset(X, y)
        return X_train, y_train, tf_dataset
    elif evaluation: 
        X_train, y_train, tf_dataset = create_dataset(X, y)
        return X_train, y_train, tf_dataset
        
