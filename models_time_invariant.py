"""
This module contains several time-invariant models.

I'm assuming a raw-audio input, which is converted to melspectrogram using Kapre.

"""
from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten, Input, Reshape, Dropout, Permute, Lambda, MaxPooling1D, Convolution1D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.wrappers import TimeDistributed
from keras import backend as K

from kapre.time_frequency import Melspectrogram
from global_config import *


def model_multi_kernel_shape(n_out, input_shape=INPUT_SHAPE,
                             out_activation='softmax'):
    """

    Symbolic summary:
    > c2' - p2 - c2 - p2 - c2 - p2 - c2 - p3 - d1
    where c2' -> multiple kernel shapes

    Parameters
    ----------
        n_out: integer, number of output nodes
        input_shape: tuple, an input shape, which doesn't include batch-axis.
        out_activation: activation function on the output
    """
    audio_input = Input(shape=input_shape)

    x = Melspectrogram(sr=SR, n_mels=64, power_melgram=2.0, return_decibel_melgram=True)(audio_input)
    x = BatchNormalization(axis=channel_axis)(x)

    x1 = Conv2D(7, (20, 3), padding='same')(x)
    x2 = Conv2D(7, (3, 3), padding='same')(x)
    x3 = Conv2D(7, (3, 20), padding='same')(x)

    x = Concatenate(axis=channel_axis)([x1, x2, x3])

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(21, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(21, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(21, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((4, 4), padding='same')(x)

    x = GlobalAveragePooling2D()(x)

    out = Dense(n_out, activation=out_activation)(x)

    model = Model(audio_input, out)

    return model


def model_crnn_icassp2017_choi(n_out, input_shape=INPUT_SHAPE,
                               out_activation='softmax'):
    """A simplified model of 
    Convolutional Recurrent Neural Networks for Music Classification,
    K Choi, G Fazekas, M Sandler, K Choi, ICASSP, 2017, New Orleans, USA

    Symbolic summary:
    > c2 - p2 - c2 - p2 - c2 - p2 - c2 - p2 - r1 - r2 - d1

    Parameters
    ----------
        n_out: integer, number of output nodes
        input_shape: tuple, an input shape, which doesn't include batch-axis.
        out_activation: activation function on the output

    """

    audio_input = Input(shape=input_shape)

    x = Melspectrogram(sr=SR, n_mels=64, power_melgram=2.0, return_decibel_melgram=True)(audio_input)
    x = BatchNormalization(axis=channel_axis)(x)

    x = Conv2D(21, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(21, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(21, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(21, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((4, 4), padding='same')(x)

    if K.image_dim_ordering() == 'channels_first':
        x = Permute((3, 1, 2))(x)

    x = Reshape((-1, 21))(x)

    # GRU block 1, 2, output
    x = GRU(41, return_sequences=True, name='gru1')(x)
    x = GRU(41, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3)(x)

    out = Dense(n_out, activation=out_activation)(x)

    model = Model(audio_input, out)

    return model


def model_conv3x3_ismir2016_choi(n_out, input_shape=INPUT_SHAPE,
                                 out_activation='softmax'):
    """ A simplified model of 
    Automatic Tagging Using Deep Convolutional Neural Networks,
    K Choi, G Fazekas, M Sandler, ISMIR, 2016, New York, USA

    Symbolic summary:
    > c2 - p2 - c2 - p2 - c2 - p2 - c2 - p2 - c2 - p3 - d1

    Modifications: 
        * n_mels (96 -> 32)
        * n_channels (many -> [16, 24, 32, 40, 48])
        * remove dropout
        * maxpooling (irregular to fit the size -> all (2, 2))
        * add GlobalAveragePooling2D
    """

    model = Sequential()
    model.add(Melspectrogram(sr=SR, n_mels=64, power_melgram=2.0, return_decibel_melgram=True,
                             input_shape=input_shape))
    model.add(BatchNormalization(axis=channel_axis))

    model.add(Conv2D(10, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(15, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(15, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(20, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(20, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(n_out, activation=out_activation))

    return model


def model_conv1d_icassp2014_sander(n_out, input_shape=INPUT_SHAPE,
                                   out_activation='softmax'):
    """A simplified model of
    End-to-end learning for music audio,
    Sander Dieleman and Benjamin Schrauwen, ICASSP, 2014

    Symbolic summary:
    > c1 - p1 - c1 - p1 - c1 - p1 - p3 - d1

    Modifications: 
        * Add BatchNormalization
        * n_mels (128 -> 32)
        * n_layers (2 -> 3)
        * add GlobalAveragePooling2D

    Parameters
    ----------
        n_out: integer, number of output nodes
        input_shape: tuple, an input shape, which doesn't include batch-axis.
        out_activation: activation function on the output

    """

    model = Sequential()
    model.add(Melspectrogram(sr=SR, n_mels=64, power_melgram=2.0, return_decibel_melgram=True,
                             input_shape=input_shape))

    model.add(Conv2D(30, (32, 4), padding='valid'))  # (None, 16, 1, N)
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), padding='same'))

    model.add(Conv2D(30, (1, 4), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), padding='same'))

    model.add(Conv2D(30, (1, 4), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), padding='same'))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(n_out, activation=out_activation))

    return model
  
def model_lstm_time_distributed(n_out, input_shape=INPUT_SHAPE):
    """ Convolutional-Recurrent Neural Networks for Live Music Genre Recognition
    Piotr Kozakowski, Jakub Królak, Łukasz Margas and Bartosz Michalak. 
    Braincode 2016 hackathon in Warsaw.
    A time_invariant model that can be used also for predicting
    on smaller audio windows (with the output_realtime layer). Built for GTZAN Genres.
    Adapted for FMA genre classification.
    
    Conv1D layers are more suitable for changes across time - they look at a small period of time as a whole, 
    extract the most valuable information and create a feature map that is still a sequence over time. 
    The features are translation-invariant only in time domain (Conv2D dont seem suitable for this)- 
    we still need to distinguish between higher and lower frequencies. 
    After each layer we use ReLU activation and 1-D max pooling, which are a pretty safe and reasonable choices.
    
    The resulting sequence of features is then fed to an LSTM layer, which should "find" both dependencies across short period of time, 
    and a long term structure of a song.
    LSTM are used since audio is a pretty long sequence in which every timestep strongly relies on both 
    the immediate predecessors and long term structure of a whole song. 
    
    After the LSTM, all the input goes into a time-distributed fully connected layer with softmax activation, essentially giving
    a sequence of N-dimensional vectors (N = number of genres) for each timestep. These vectors represent the network’s 
    belief of the music genre at the particular point of time, modelled as probability distributions.
    
    In the end we take an arithmetic mean across time of all the predicted distributions and return it as a final answer. 
    These mean of vectors is too a valid distribution.
    
    Modifications: 
      * Added Melspectrogram
      * CONV_FILTER_COUNT (256 -> 32)
      * LSTM_COUNT (256 -> 64)
      * Changed DROPOUT layers with LSTM internal dropout
    
    Symbolic summary:
    > c1 - p1 - c1 - p1 - c1 - p1 - r1 - r2 - d1
    
    Summary:
      input (InputLayer)           (None, 1, 160000)         0         
      _________________________________________________________________
      melspectrogram_2 (Melspectro (None, 1, 128, 625)       296064    
      _________________________________________________________________
      batch_normalization_2 (Batch (None, 1, 128, 625)       4         
      _________________________________________________________________
      permute_2 (Permute)          (None, 1, 625, 128)       0         
      _________________________________________________________________
      lambda_2 (Lambda)            (None, 625, 128)          0         
      _________________________________________________________________
      convolution_1 (Conv1D)       (None, 621, 32)           20512     
      _________________________________________________________________
      activation_4 (Activation)    (None, 621, 32)           0         
      _________________________________________________________________
      max_pooling1d_4 (MaxPooling1 (None, 310, 32)           0         
      _________________________________________________________________
      convolution_2 (Conv1D)       (None, 306, 32)           5152      
      _________________________________________________________________
      activation_5 (Activation)    (None, 306, 32)           0         
      _________________________________________________________________
      max_pooling1d_5 (MaxPooling1 (None, 153, 32)           0         
      _________________________________________________________________
      convolution_3 (Conv1D)       (None, 149, 32)           5152      
      _________________________________________________________________
      activation_6 (Activation)    (None, 149, 32)           0         
      _________________________________________________________________
      max_pooling1d_6 (MaxPooling1 (None, 74, 32)            0         
      _________________________________________________________________
      lstm_2 (LSTM)                (None, 74, 64)            24832     
      _________________________________________________________________
      time_distributed_2 (TimeDist (None, 74, 8)             520       
      _________________________________________________________________
      output_realtime (Activation) (None, 74, 8)             0         
      _________________________________________________________________
      output_merged (Lambda)       (None, 8)                 0         
      =================================================================
      Total params: 352,236
      Trainable params: 56,170
      Non-trainable params: 296,066
      _________________________________________________________________
        
    Parameters
    ----------
        n_out: integer, number of output nodes
        input_shape: tuple, an input shape, which doesn't include batch-axis.
        out_activation: activation function on the output

    """
    
    N_LAYERS = 3
    FILTER_LENGTH = 5
    CONV_FILTER_COUNT = 32
    LSTM_COUNT = 64

    model_input = Input(input_shape, name='input')
    layer = Melspectrogram(sr=SR, n_mels=128, power_melgram=2.0,
                           return_decibel_melgram=True)(model_input)

    layer = BatchNormalization(axis=channel_axis)(layer)

    if K.image_data_format() == 'channels_first':  # (ch, freq, time)
        layer = Permute((1, 3, 2))(layer)

    layer = Lambda(lambda x: K.squeeze(x, axis=1),
                   output_shape=lambda shape: (shape[0],) + shape[2:])(layer)

    for i in range(N_LAYERS):
        layer = Convolution1D(
            filters=CONV_FILTER_COUNT,
            kernel_size=FILTER_LENGTH,
            name='convolution_' + str(i + 1)
        )(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling1D(2)(layer)

    # layer = Dropout(0.5)(layer)
    layer = LSTM(LSTM_COUNT, return_sequences=True, recurrent_dropout=0.25, dropout=0.25)(layer)
    # layer = Dropout(0.5)(layer)
    layer = TimeDistributed(Dense(n_out))(layer)
    layer = Activation('softmax', name='output_realtime')(layer)
    time_distributed_merge_layer = Lambda(
        function=lambda x: K.mean(x, axis=1),
        output_shape=lambda shape: (shape[0],) + shape[2:],
        name='output_merged'
    )
    model_output = time_distributed_merge_layer(layer)
    model = Model(model_input, model_output)

    return model


if __name__ == "__main__":
    model = model_multi_kernel_shape(8)
    model.summary()
