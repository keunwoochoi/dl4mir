"""
This module contains several time-varying models.

I'm assuming a raw-audio input, which is converted to melspectrogram using Kapre.

"""
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Input, Reshape, Permute, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras import backend as K

from kapre.time_frequency import Melspectrogram

from global_config import *


def model_convrnn(n_out, input_shape=(1, None), out_activation='softmax'):
    """No reference, just ConvRNN.

    Symbolic summary:
    > c2 - c2 - c2 - c2 - r2 - r2 - d1

    Parameters
    ----------
        n_out: integer, number of output nodes
        input_shape: tuple, an input shape, which doesn't include batch-axis.
                     (1, None) means (mono channel, variable length).
        out_activation: activation function on the output

    """
    assert input_shape[0] == 1, 'Mono input please!'
    model = Sequential()
    n_mels = 64
    model.add(Melspectrogram(sr=SR, n_mels=n_mels, power_melgram=2.0,
                             return_decibel_melgram=True,
                             input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    model.add(Conv2D(1, (1, 1), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))

    if K.image_dim_ordering() == 'channels_first':  # (ch, freq, time)
        model.add(Permute((3, 2, 1)))  # (time, freq, ch)
    else:  # (freq, time, ch)
        model.add(Permute((2, 1, 3)))  # (time, ch, freq)

    # model.add(Reshape((-1, n_mels * n_ch))) # (time, ch * freq)
    # Reshape for LSTM
    model.add(Lambda(lambda x: K.squeeze(x, axis=3),
                     output_shape=squeeze_output_shape))

    model.add(LSTM(25, return_sequences=True))
    model.add(LSTM(25, return_sequences=True))

    model.add(TimeDistributed(Dense(n_out, activation=out_activation)))

    return model


def model_lstm_leglaive_icassp2014(n_out, input_shape=(1, None),
                                   out_activation='softmax', bidirectional=True):
    """Singing voice detection with deep recurrent neural networks
    Simon Leglaive, Romain Hennequin, Roland Badeau, ICASSP 2015

    Symbolic summary:
    > bi_r1 - bi_r1 - bi_r1 -
    > r1 - r1 - r1 - d1

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
    model.add(Lambda(lambda x: K.squeeze(x, axis=3),
                     output_shape=squeeze_output_shape))
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


def squeeze_output_shape(input_shape):
    return input_shape[:3]


if __name__ == '__main__':
    model = model_lstm_leglaive_icassp2014(2)
    model.summary()

    model = model_lstm_leglaive_icassp2014(2, bidirectional=False)
    model.summary()

    model = model_convrnn(2, input_shape=INPUT_SHAPE)
    model.summary()

    model = model_convrnn(2, input_shape=(1, None))
    model.summary()
