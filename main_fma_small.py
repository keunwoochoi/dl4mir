""" 16 May 2017, Keunwoo Choi (keunwoo.choi@qmul.ac.uk)

It assumes FMA-small dataset is downloaded and pre-processed by main_preprocess.py.
3
"""
from __future__ import print_function  # (at top of module)

import os
import pandas as pd
import numpy as np
import models_excerpt
import models_MLP

from sklearn.preprocessing import LabelEncoder

import utils_preprocess
import my_callbacks
from global_config import *


# TODO: add MLP models.

def data_gen(df_subset, ys, is_shuffle, batch_size=40):
    """Data generator.

    df_subset: pandas dataframe, with rows subset
    ys: numpy arrays, N-by-8 one-hot-encoded labels
    is_shuffle: shuffle every batch if True.
    batch_size: integer, size of batch. len(df_subset) % batch_size should be 0.
    """
    n_data = len(df_subset)
    n_batch = n_data // batch_size
    if n_data % batch_size != 0:
        print("= WARNING =")
        print("  n_data % batch_size != 0 but this code does not assume it")
        print("  so the residual {} sample(s) will be ignored.".format(n_data % batch_size))

    while True:
        for batch_i in xrange(n_batch):
            if is_shuffle:
                batch_idxs = np.random.choice(n_data, batch_size, replace=False)
            else:
                batch_idxs = range(batch_i * batch_size, (batch_i + 1) * batch_size)

            src_batch = np.array([utils_preprocess.load_npy_fma(df_subset.index[i]) for i in batch_idxs],
                                 dtype=K.floatx())
            src_batch = src_batch[:, np.newaxis, :]  # make (batch, N) to (batch, 1, N) for kapre compatible

            y_batch = np.array([ys[i] for i in batch_idxs],
                               dtype=K.floatx())
            yield src_batch, y_batch


def main(model_name, exp_name='fma'):
    """
    DO it!
    """
    assert model_name in ['multi_kernel', 'crnn', 'cnn3x3', 'cnn1d']
    print("-" * 60)
    print("Keunwoo: Welcome! Lets do something deep with FMA dataset.")
    print("         I'm assuming you finished pre-processing.")
    print("         We're gonna use {} model".format(model_name))
    csv_path = os.path.join(DIR_FMA_CSV, 'tracks.csv')

    tracks = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    small = tracks['set', 'subset'] == 'small'
    training = (tracks['set', 'split'] == 'training') & small
    validation = (tracks['set', 'split'] == 'validation') & small
    test = (tracks['set', 'split'] == 'test') & small

    print("Keunwoo: We're loading and modifying label values.")
    enc = LabelEncoder()
    y_train = enc.fit_transform(tracks[training]['track', 'genre_top'])
    y_valid = enc.transform(tracks[validation]['track', 'genre_top'])
    y_test = enc.transform(tracks[test]['track', 'genre_top'])

    y_train = np.eye(8)[y_train]
    y_valid = np.eye(8)[y_valid]
    y_test = np.eye(8)[y_test]

    print("It's a good practice to use callbacks in Keras.")
    callbacks = my_callbacks.get_callbacks(name=exp_name)
    early_stopper, model_saver, weight_saver, csv_logger = callbacks

    print("Preparing data generators for training and validation...")
    batch_size = 40
    steps_per_epoch = len(y_train) // batch_size
    gen_train = data_gen(tracks[training], y_train, True, batch_size=batch_size)
    gen_valid = data_gen(tracks[validation], y_valid, False, batch_size=batch_size)

    print("Keunwoo: Getting model...")
    if model_name == 'multi_kernel':
        model = models_excerpt.model_multi_kernel_shape(n_out=8)
    elif model_name == 'crnn':
        model = models_excerpt.model_crnn_icassp2017_choi(n_out=8)
    elif model_name == 'cnn3x3':
        model = models_excerpt.model_conv3x3_ismir2016_choi(n_out=8)
    elif model_name == 'cnn1d':
        model = models_excerpt.model_conv1d_icassp2014_sander(n_out=8)

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    print("Keunwoo: Starting to train...")
    model.fit_generator(gen_train, steps_per_epoch, epochs=5,
                        callbacks=callbacks,
                        validation_data=gen_valid,
                        validation_steps=len(y_valid) // batch_size)

    print("Keunwoo: Training is done. Loading the best weights...")
    model.load_weights("{}_best_weights.h5".format(exp_name))
    gen_test = data_gen(tracks[test], y_test, False, batch_size=batch_size)

    print("         Evaluating...")
    scores = model.evaluate_generator(gen_test, len(y_test) // batch_size)

    print('Keunwoo: Done for {}!'.format(model_name))
    print("         test set loss:{}".format(scores[0]))
    print("         test set accuracy: {}%".format([scores[1]]))


if __name__ == '__main__':
    main('multi_kernel', 'fma_multi_kernel')
    main('crnn', 'fma_crnn')
    main('cnn3x3', 'fma_cnn3x3')
    main('cnn1d', 'fma_cnn1d')

"""
Keunwoo: Welcome! Lets do something deep with FMA dataset.
         I'm assuming you finished pre-processing.
         We're gonna use multi_kernel model
Keunwoo: We're loading and modifying label values.
It's a good practice to use callbacks in Keras.
Preparing data generators for training and validation...
Keunwoo: Getting model...
Keunwoo: Starting to train...

Epoch 1/5
160/160 [==============================] - 53s - loss: 1.7421 - acc: 0.3731 - val_loss: 2.0509 - val_acc: 0.2350
Epoch 2/5
160/160 [==============================] - 48s - loss: 1.5444 - acc: 0.4503 - val_loss: 1.8888 - val_acc: 0.2775
Epoch 3/5
160/160 [==============================] - 48s - loss: 1.4819 - acc: 0.4730 - val_loss: 1.6285 - val_acc: 0.4063
Epoch 4/5
160/160 [==============================] - 48s - loss: 1.4061 - acc: 0.5120 - val_loss: 1.6534 - val_acc: 0.3988
Epoch 5/5
160/160 [==============================] - 48s - loss: 1.3584 - acc: 0.5248 - val_loss: 1.5023 - val_acc: 0.4938
Keunwoo: Training is done. Loading the best weights...
         Evaluating...
Keunwoo: Done for multi_kernel!
         test set loss:1.5535479486
         test set accuracy: [0.47750000953674315]%
------------------------------------------------------------
Keunwoo: Welcome! Lets do something deep with FMA dataset.
         I'm assuming you finished pre-processing.
         We're gonna use crnn model
Keunwoo: We're loading and modifying label values.
It's a good practice to use callbacks in Keras.
Preparing data generators for training and validation...
Keunwoo: Getting model...
Keunwoo: Starting to train...
Epoch 1/5
160/160 [==============================] - 63s - loss: 1.8359 - acc: 0.2875 - val_loss: 2.2506 - val_acc: 0.1900
Epoch 2/5
160/160 [==============================] - 61s - loss: 1.6170 - acc: 0.4120 - val_loss: 2.3764 - val_acc: 0.2500
Epoch 3/5
160/160 [==============================] - 60s - loss: 1.5181 - acc: 0.4548 - val_loss: 1.8484 - val_acc: 0.3513
Epoch 4/5
160/160 [==============================] - 60s - loss: 1.4217 - acc: 0.4989 - val_loss: 1.5957 - val_acc: 0.4375
Epoch 5/5
160/160 [==============================] - 60s - loss: 1.3699 - acc: 0.5247 - val_loss: 1.5935 - val_acc: 0.4250
Keunwoo: Training is done. Loading the best weights...
         Evaluating...
Keunwoo: Done for crnn!
         test set loss:1.67180262804
         test set accuracy: [0.39250000640749932]%
------------------------------------------------------------
Keunwoo: Welcome! Lets do something deep with FMA dataset.
         I'm assuming you finished pre-processing.
         We're gonna use cnn3x3 model
Keunwoo: We're loading and modifying label values.
It's a good practice to use callbacks in Keras.
Preparing data generators for training and validation...
Keunwoo: Getting model...
Keunwoo: Starting to train...
Epoch 1/5
160/160 [==============================] - 30s - loss: 1.7715 - acc: 0.3659 - val_loss: 2.0864 - val_acc: 0.2013
Epoch 2/5
160/160 [==============================] - 29s - loss: 1.5308 - acc: 0.4581 - val_loss: 1.7897 - val_acc: 0.3275
Epoch 3/5
160/160 [==============================] - 29s - loss: 1.4375 - acc: 0.5031 - val_loss: 1.6883 - val_acc: 0.3975
Epoch 4/5
160/160 [==============================] - 29s - loss: 1.3595 - acc: 0.5261 - val_loss: 2.0865 - val_acc: 0.3238
Epoch 5/5
160/160 [==============================] - 29s - loss: 1.3319 - acc: 0.5344 - val_loss: 2.0630 - val_acc: 0.3263
Keunwoo: Training is done. Loading the best weights...
         Evaluating...
Keunwoo: Done for cnn3x3!
         test set loss:1.86332789063
         test set accuracy: [0.3225000061094761]%
------------------------------------------------------------
Keunwoo: Welcome! Lets do something deep with FMA dataset.
         I'm assuming you finished pre-processing.
         We're gonna use cnn1d model
Keunwoo: We're loading and modifying label values.
It's a good practice to use callbacks in Keras.
Preparing data generators for training and validation...
Keunwoo: Getting model...
Keunwoo: Starting to train...
Epoch 1/5
160/160 [==============================] - 31s - loss: 1.7900 - acc: 0.3372 - val_loss: 2.5954 - val_acc: 0.1300
Epoch 2/5
160/160 [==============================] - 29s - loss: 1.6272 - acc: 0.4186 - val_loss: 1.9440 - val_acc: 0.2625
Epoch 3/5
160/160 [==============================] - 28s - loss: 1.5502 - acc: 0.4498 - val_loss: 3.3067 - val_acc: 0.1913
Epoch 4/5
160/160 [==============================] - 28s - loss: 1.5200 - acc: 0.4548 - val_loss: 2.0652 - val_acc: 0.2288
Epoch 5/5
160/160 [==============================] - 28s - loss: 1.4641 - acc: 0.4764 - val_loss: 3.2042 - val_acc: 0.2100
Keunwoo: Training is done. Loading the best weights...
         Evaluating...
Keunwoo: Done for cnn1d!
         test set loss:1.92252177596
         test set accuracy: [0.23250000523403286]%

"""
