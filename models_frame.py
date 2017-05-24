from __future__ import print_function    # (at top of module)

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten, Input, Reshape, Dropout, Permute
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras import backend as K

from kapre.time_frequency import Melspectrogram

from global_config import *


def model_convrnn(n_out, input_shape=INPUT_SHAPE, out_activation='softmax'):
    """No reference, just ConvRNN.


    Parameters
    ----------
        n_out: integer, number of output nodes
        input_shape: tuple, an input shape, which doesn't include batch-axis.
        out_activation: activation function on the output

    """
    assert input_shape[0] == 1, 'Mono input please!'
    model = Sequential()
    model.add(Melspectrogram(sr=SR, n_mels=40, power_melgram=2.0,
                             return_decibel_melgram=True,
                             input_shape=input_shape))
    model.add(Conv2D(30, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(30, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    if K.image_dim_ordering() == 'channels_first':
        model.add(Permute((3, 1, 2)))

    model.add(Reshape((-1, 30)))

    model.add(LSTM(25, return_sequences=True))
    model.add(LSTM(25, return_sequences=True))

    model.add(TimeDistributed(Dense(n_out, activation=out_activation)))

    return model


def model_lstm_leglaive_icassp2014(n_out, input_shape=INPUT_SHAPE,
                                   out_activation='softmax', bidirectional=True):
    """Singing voice detection with deep recurrent neural networks
    Simon Leglaive, Romain Hennequin, Roland Badeau, ICASSP 2015

    Parameters
    ----------
        n_out: integer, number of output nodes
        input_shape: tuple, an input shape, which doesn't include batch-axis.
        out_activation: activation function on the output
        bidirectional: boolean, to specify whether rnn is bidirectional or not.

    """
    assert input_shape[0] == 1, 'Mono input please!'
    model = Sequential()
    model.add(Melspectrogram(sr=SR, n_mels=40, power_melgram=2.0,
                             return_decibel_melgram=True,
                             input_shape=input_shape))

    if K.image_data_format() == 'channels_first':
        model.add(Permute((3, 2, 1)))  # ch, freq, time -> time, freq, ch
    else:
        model.add(Permute((2, 1, 3)))  # freq, time, ch -> time, freq, ch

    model.add(BatchNormalization(axis=channel_axis))

    # Reshape for LSTM
    output_shape = K.int_shape(model.output)
    model.add(Reshape((output_shape[1], output_shape[2])))
    if bidirectional:
        # Use Bidirectional LSTM
        model.add(Bidirectional(LSTM(30, return_sequences=True)))
        model.add(Bidirectional(LSTM(20, return_sequences=True)))
        model.add(Bidirectional(LSTM(40, return_sequences=True)))
    else:
        # Use normal LSTM
        model.add(LSTM(30 * 2, return_sequences=True))
        model.add(LSTM(20 * 2, return_sequences=True))
        model.add(LSTM(40 * 2, return_sequences=True))

    model.add(TimeDistributed(Dense(n_out, activation=out_activation)))

    return model


if __name__ == '__main__':
    model = model_lstm_leglaive_icassp2014(2)
    model.summary()

    model = model_lstm_leglaive_icassp2014(2, bidirectional=False)
    model.summary()

    model = model_convrnn(2)
    model.summary()
