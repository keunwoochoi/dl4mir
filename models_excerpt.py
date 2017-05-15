from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten, Input, Reshape, Dropout, Permute
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras import backend as K

from kapre.time_frequency import Melspectrogram
from global_config import *

def model_multi_kernel_shape(n_out, input_shape=INPUT_SHAPE):
    audio_input = Input(shape=input_shape)

    x = Melspectrogram(sr=SR, n_mels=32, power_melgram=2.0, return_decibel_melgram=True)(audio_input)
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

    out = Dense(n_out)(x)

    model = Model(audio_input, out)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_crnn_icassp2017_choi(n_out, input_shape=INPUT_SHAPE):
    """A simplified model of 
    Convolutional Recurrent Neural Networks for Music Classification,
    K Choi, G Fazekas, M Sandler, K Choi, ICASSP, 2017, New Orleans, USA

    """

    audio_input = Input(shape=input_shape)

    x = Melspectrogram(sr=SR, n_mels=32, power_melgram=2.0, return_decibel_melgram=True)(audio_input)
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

    out = Dense(n_out)(x)

    model = Model(audio_input, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_conv3x3_ismir2016_choi(n_out, input_shape=INPUT_SHAPE):
    """ A simplified model of 
    Automatic Tagging Using Deep Convolutional Neural Networks,
    K Choi, G Fazekas, M Sandler, ISMIR, 2016, New York, USA

    Modifications: 
        * n_mels (96 -> 32)
        * n_channels (many -> [16, 24, 32, 40, 48])
        * remove dropout
        * maxpooling (irregular to fit the size -> all (2, 2))
        * add GlobalAveragePooling2D
    """

    model = Sequential()
    model.add(Melspectrogram(sr=SR, n_mels=32, power_melgram=2.0, return_decibel_melgram=True,
                             input_shape=input_shape))
    model.add(BatchNormalization(axis=channel_axis))

    model.add(Conv2D(20, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(25, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(30, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(35, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(40, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(n_out))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def model_conv1d_icassp2014_sander(n_out, input_shape=INPUT_SHAPE):
    """ A simplified model of
    End-to-end learning for music audio,
    Sander Dieleman and Benjamin Schrauwen, ICASSP, 2014
    
    Modifications: 
        * Add BatchNormalization
        * n_mels (128 -> 32)
        * n_layers (2 -> 3)
        * add GlobalAveragePooling2D

    """

    model = Sequential()
    model.add(Melspectrogram(sr=SR, n_mels=32, power_melgram=2.0, return_decibel_melgram=True,
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

    model.add(Dense(n_out))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":

    model = model_multi_kernel_shape(8)
    model.summary()
