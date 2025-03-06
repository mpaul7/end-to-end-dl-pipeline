import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_synthetic_data(num_samples=1000, time_steps=10, num_features=2):
    X = np.random.rand(num_samples, time_steps, num_features)
    y = np.random.randint(0, 2, size=(num_samples, 1))
    return X, y

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

def build_model(hp, input_shape):
    model = Sequential()
    
    # First LSTM layer with tunable units
    model.add(LSTM(
        units=get_hp_value(hp, 'units_lstm1', hp_config['units_lstm1']),
        return_sequences=True,
        input_shape=input_shape
    ))
    model.add(Dropout(get_hp_value(hp, 'dropout_rate', hp_config['dropout_rate'])))

    # Second LSTM layer with tunable units
    model.add(LSTM(units=get_hp_value(hp, 'units_lstm2', hp_config['units_lstm2'])))
    model.add(Dropout(get_hp_value(hp, 'dropout_rate', hp_config['dropout_rate'])))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with tunable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
        learning_rate=get_hp_value(hp, 'learning_rate', hp_config['learning_rate'])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Generate training and testing data
    X_train, y_train = generate_synthetic_data(num_samples=800)
    X_test, y_test = generate_synthetic_data(num_samples=200)

    # Create model builder with input shape
    model_builder = lambda hp: build_model(hp, (X_train.shape[1], X_train.shape[2]))

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
