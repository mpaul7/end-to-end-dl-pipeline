import tensorflow as tf
from keras import layers

KERAS_INITIALIZER = {
    'default': tf.keras.initializers.GlorotUniform,
    'he_uniform': tf.keras.initializers.HeUniform,
    'he_normal': tf.keras.initializers.HeNormal
}


def create_dl_model_cnn(params):

    if params['regularizer'] == 'l1':
            regularizer = tf.keras.regularizers.L1(params['regularizer_value'])
    elif params['regularizer'] == 'l2':
            regularizer = tf.keras.regularizers.L2(params['regularizer_value'])
    else:
            regularizer = None

    """Create input layers for packet sequence data """
    
    inputs = {name: layers.Input(shape=(params['cnn_stat_feature_length'],), dtype=tf.float32, name=name) for name in params['cnn_stat_feature']}
    pktseq_x = tf.stack(list(inputs.values()), axis=2)
    # pktseq_x = layers.Conv1D(200, kernel_size=7, strides=1, kernel_regularizer=regularizer,  padding='same', input_shape=(None, 3))(pktseq_x)
    # pktseq_x = layers.ReLU()(pktseq_x)
    # pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    # pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)


    flow_inputs = {name: layers.Input(shape=(1,), dtype=tf.float32, name=name) for name in params.features}

    flow_x = layers.Concatenate(axis=-1)(list(flow_inputs.values()))
    flow_x = layers.Reshape(target_shape=(len(params.features),))(flow_x)

    flow_x = preprocessor_flow(flow_x)
    flow_x = layers.Reshape(target_shape=(len(params.features),1))(flow_x)

    flow_x = layers.Conv1D(64, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='same', input_shape=(None, 1))(flow_x)
    flow_x = layers.ReLU()(flow_x)
    flow_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(flow_x)
    # flow_x = layers.Dropout(params.dropout_rate)(flow_x)

    flow_x = layers.Conv1D(64, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='same')(flow_x)
    flow_x = layers.ReLU()(flow_x)
    flow_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(flow_x)
    # flow_x = layers.Dropout(params.dropout_rate)(flow_x)

    flow_x = layers.Conv1D(64, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='same')(flow_x)
    flow_x = layers.ReLU()(flow_x)
    # flow_x = layers.Dropout(params.dropout_rate)(flow_x)
    flow_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(flow_x)

    flow_x = layers.Conv1D(64, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='same')(flow_x)
    flow_x = layers.ReLU()(flow_x)
    # flow_x = layers.Dropout(params.dropout_rate)(flow_x)
    flow_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(flow_x)

    flow_x = layers.Conv1D(64, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='valid')(flow_x)
    flow_x = layers.ReLU()(flow_x)
    flow_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(flow_x)
    # flow_x = layers.Dropout(params.dropout_rate)(flow_x)

    flow_x = layers.Conv1D(96, kernel_size=5, strides=1, kernel_regularizer=regularizer, padding='valid')(flow_x)
    flow_x = layers.ReLU()(flow_x)
    flow_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(flow_x)
    # flow_x = layers.Dropout(params.dropout_rate)(flow_x)

    flow_x = layers.Conv1D(96, kernel_size=5, strides=2, kernel_regularizer=regularizer, padding='valid')(flow_x)
    flow_x = layers.ReLU()(flow_x)
    flow_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(flow_x)
    flow_x = layers.Dropout(params.dropout_rate)(flow_x)

    flow_x = layers.GlobalAveragePooling1D()(flow_x)
    flow_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(flow_x)
    flow_x = layers.Dropout(0.5)(flow_x)

    # input_branches['inputs'].append(flow_inputs)
    # input_branches['layer'].append(flow_x)
    
    """Output layer"""
    # outputs = layers.Dense(output_units, activation='softmax', name='softmax')(pktseq_x)
    # model = models.Model(inputs=[inputs], outputs=outputs)

    return flow_inputs, flow_x