from pathlib import Path
import json
import keras_tuner as kt
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

from src.dlProject.utils.features import stat_features_tr, target_labels    
from src.dlProject.commons.create_tf_dataset_for_hp import create_train_test_dataset_tf

# Set seed for reproducibility
tf.random.set_seed(42)


# Define hyperparameter configuration
hp_config = {
    'units_lstm1': {
        'min_value': 32,
        'max_value': 128,
        'step': 32
    },
    'units_lstm2': {
        'min_value': 16,
        'max_value': 64,
        'step': 16
    },
    'dropout_rate': {
        'min_value': 0.1,
        'max_value': 0.5,
        'step': 0.1
    },
    'learning_rate': {
        'values': [1e-2, 1e-3, 1e-4]
    },
    'batch_size': {
        'values': [16, 32, 64]
    }
}

def get_hp_value(hp, param_name, param_config):
    if 'values' in param_config:
        return hp.Choice(param_name, values=param_config['values'])
    else:
        return hp.Int(param_name, 
                     min_value=param_config['min_value'],
                     max_value=param_config['max_value'], 
                     step=param_config['step']) if isinstance(param_config['min_value'], int) else \
               hp.Float(param_name,
                       min_value=param_config['min_value'],
                       max_value=param_config['max_value'],
                       step=param_config['step'])

def build_model(hp):
    regularizer = tf.keras.regularizers.L1(0.1)
        
    inputs = {name: layers.Input(shape=(87,), dtype=tf.float32, name=name) for name in stat_features_tr}

    """Stack input layers"""
    pktseq_x = tf.stack(list(inputs.values()), axis=2)

    pktseq_x = layers.Conv1D(64, kernel_size=3, strides=1, kernel_regularizer=regularizer,  padding='same', input_shape=(None, 3))(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(0.1)(pktseq_x)

    pktseq_x = layers.Conv1D(64, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(0.1)(pktseq_x)

    pktseq_x = layers.Conv1D(64, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(0.1)(pktseq_x)

    pktseq_x = layers.Conv1D(64, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(0.1)(pktseq_x)

    pktseq_x = layers.Conv1D(64, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(0.1)(pktseq_x)
# =========
    pktseq_x = layers.Conv1D(96, kernel_size=5, strides=1, kernel_regularizer=regularizer, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(0.2)(pktseq_x)

    pktseq_x = layers.Conv1D(96, kernel_size=5, strides=2, kernel_regularizer=regularizer, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    # pktseq_x = layers.Dropout(params.model_params.dropout_rate)(pktseq_x)
    pktseq_x = layers.Dropout(0.2)(pktseq_x)

    pktseq_x = layers.GlobalAveragePooling1D()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    # pktseq_x = layers.Dropout(0.5)(pktseq_x)


    input_branches = {'inputs': [], 'layer': []}
    input_branches['inputs'].append(inputs)
    input_branches['layer'].append(pktseq_x)  

    # Combine branches if multiple
    if len(input_branches['layer']) > 1:
        x = tf.keras.layers.concatenate(input_branches['layer'])
    else:
        x = input_branches['layer'][0]
    
    # Add dense layers
    num_dense = 2
    dense_units_list = [128, 64]
    for i in range(num_dense): # 2
        x = layers.Dense(units=dense_units_list[i], 
                        kernel_initializer=tf.keras.initializers.HeUniform,
                        name=f'final_dense_{i}'
                        )(x)
        x = layers.LeakyReLU(name=f'dense_leaky_relu_{i}')(x)
        x = layers.Dropout(0.5, name=f'dense_dropout_{i}')(x)
    
    # Add final dense layers
    num_final_dense = 2
    final_dense_units_list = [128, 64]
    for i in range(num_final_dense):
        x = layers.Dense(units=final_dense_units_list[i], 
                        kernel_initializer=tf.keras.initializers.HeUniform,
                        name=f'final_dense_{i}'
                        )(x)
        x = layers.LeakyReLU(name=f'final_dense_leaky_relu_{i}')(x)
        x = layers.Dropout(0.5, name=f'final_dense_dropout_{i}')(x)
    
    # Add output layer
    outputs = tf.keras.layers.Dense(44, activation='softmax', name='softmax')(x)
    
    # Create final model
    model = models.Model(inputs=input_branches['inputs'], outputs=outputs)

    # Compile the model with tunable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
        learning_rate=get_hp_value(hp, 'learning_rate', hp_config['learning_rate'])
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']    
    )  
    
    return model

def main():
    # Generate training and testing data
    with open('params.json', 'r') as f:
        params = json.load(f)

    params['features']['cnn_stat_feature'] = stat_features_tr
    print("Number of statistical features:", len(stat_features_tr))
    print(params['features']['cnn_stat_feature'])
    params['features']['target_labels'] = target_labels
    print("Number of target labels:", len(target_labels))
    print(params['features']['target_labels'])

    X, y = create_train_test_dataset_tf(
            data_file=Path('/home/solana/Downloads/data_sources_solana_train.parquet'),
            params=params,
            train=True,
            evaluation=False
        )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create model builder with input shape
    model_builder = lambda hp: build_model(hp)

    # Initialize Keras Tuner
    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory='tuner_dir',
        project_name='lstm_hyperparameter_tuning'
    )

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Search for the best hyperparameters
    tuner.search(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        callbacks=[early_stopping]
    )

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Print best hyperparameters
    print("Best Hyperparameters:")
    for param, config in hp_config.items():
        print(f" - {param}: {best_hps.get(param)}")

    # Build and train the best model
    batch_size = best_hps.get('batch_size', 32)
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Log results with MLflow
    mlflow.set_experiment("LSTM-Hyperparameter-Tuning-Demo")
    
    with mlflow.start_run():
        # Log hyperparameters
        for param, config in hp_config.items():
            mlflow.log_param(param, best_hps.get(param))
        mlflow.log_param("epochs", 20)

        # Log metrics
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)

        # Log model with signature
        input_example = X_test[:1]
        signature = infer_signature(X_test, best_model.predict(X_test))
        
        mlflow.keras.log_model(
            best_model,
            artifact_path="best_lstm_model",
            input_example=input_example,
            signature=signature
        )

        print("Best model logged to MLflow")

if __name__ == "__main__":
    main()
