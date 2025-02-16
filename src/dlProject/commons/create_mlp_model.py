import tensorflow as tf
from keras import layers

KERAS_INITIALIZER = {
        'default': tf.keras.initializers.GlorotUniform,
        'he_uniform': tf.keras.initializers.HeUniform,
        'he_normal': tf.keras.initializers.HeNormal
}

"""Create Keras DL model - MLP"""
def create_dl_model_mlp(params):
        
    if params.model_params.regularizer == 'l1':
        regularizer = tf.keras.regularizers.L1(params.model_params.regularizer.l1)
    elif params.model_params.regularizer == 'l2':
        regularizer = tf.keras.regularizers.L2(params.model_params.regularizer.l2)
    else:
        regularizer = None

    initializer = KERAS_INITIALIZER[params.model_params.initializer]
    
    """Create input layers for packet sequence data """
    inputs = {name: layers.Input(shape=(1,), dtype=tf.float32, name=name)for name in params.features.stat_features}

    """Stack input layers"""
    x = layers.Concatenate(axis=-1)(list(inputs.values()))
    
    """Create chain of Dense layers"""
    for i in range(params.mlp.num_dense):
        x = layers.Dense(units=params.mlp.units_list[i], kernel_regularizer=regularizer, kernel_initializer=initializer, name=f'dense_{i}')(x)
        x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
        x = layers.LeakyReLU(name=f'leaky_relu_{i}')(x)
        x = layers.Dropout(params.model_params.dropout_rate, name=f'dropout_{i}')(x)

        
    return inputs, x