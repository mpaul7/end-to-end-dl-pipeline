

import tensorflow as tf
from keras import layers

KERAS_INITIALIZER = {
    'default': tf.keras.initializers.GlorotUniform,
    'he_uniform': tf.keras.initializers.HeUniform,
    'he_normal': tf.keras.initializers.HeNormal
}

def create_dl_model_lstm(params):
    
    """Create input layers for packet sequence data """
    inputs = {name: layers.Input(shape=(params['sequence_length'],), dtype=tf.float32, name=name)for name in params['seq_packet_feature']}
    
    """Stack input layers"""
    pktseq_x1 = tf.stack(list(inputs.values()), axis=2)
    pktseq_x2 = layers.Reshape(target_shape=(params['sequence_length'], 1))(list(inputs.values())[-1])
    # pktseq_x = layers.Concatenate(axis=-1)([pktseq_x1, pktseq_x2])
    
    """LSTM units layer"""
    if params['lstm']['num_lstm'] == 1:
        if params['regularizer'] == 'l1':
            regularizer = tf.keras.regularizers.L1(params['regularizer_value'])
        elif params['regularizer'] == 'l2':
            regularizer = tf.keras.regularizers.L2(params['regularizer_value'])
        else:
            regularizer = None
        lstm = layers.LSTM(units=params['lstm']['lstm_units'], 
                            input_shape=(params['sequence_length'], 3), 
                            recurrent_dropout=params['dropout_rate'],   
                            kernel_regularizer=regularizer,
                            # recurrent_regularizer=regularizer,
                            name=f'lstm_1'
                        )(pktseq_x1)
    if params['lstm']['num_lstm'] == 2:
        if params['regularizer'] == 'l1':
            regularizer = tf.keras.regularizers.L1(params['regularizer_value'])
        elif params['regularizer'] == 'l2':
            regularizer = tf.keras.regularizers.L2(params['regularizer_value'])
        else:
            regularizer = None
        lstm = layers.LSTM(params['lstm']['lstm_units'], 
                                input_shape=(params['sequence_length'], 3),
                                return_sequences=True,
                                recurrent_dropout=params['dropout_rate'],
                                kernel_regularizer=regularizer,
                                # recurrent_regularizer=regularizer,
                                name=f'lstm_1'
                                )(pktseq_x1)
        lstm = layers.LSTM(params['lstm']['lstm_units'],    
                                input_shape=(params['sequence_length'], 3),
                                go_backwards=True, 
                                recurrent_dropout=params['dropout_rate'],
                                kernel_regularizer=regularizer,
                                # recurrent_regularizer=regularizer,
                                name=f'lstm_2'
                                )(lstm)
    """Create chain of Dense layers"""
    for i in range(params['lstm']['num_lstm_dense']): # 2
        lstm = layers.Dense(units=params['lstm']['lstm_dense_units_list'][i], 
                            kernel_initializer=KERAS_INITIALIZER[params['initializer']],
                            name=f'lstm_dense_{i}'
                            )(lstm)
        lstm = layers.LeakyReLU(name=f'lstm_leaky_relu_{i}')(lstm)
        lstm = layers.Dropout(params['dropout_rate'], name=f'lstm_dropout_{i}')(lstm)
    
    return inputs, lstm