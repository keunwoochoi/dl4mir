""" 16 May 2017, Keunwoo Choi (keunwoo.choi@qmul.ac.uk)

It assumes FMA-small dataset is downloaded and pre-processed by main_preprocess.py.

"""
from __future__ import print_function  # (at top of module)

import os
import pandas as pd
import numpy as np
import models_excerpt

from sklearn.preprocessing import LabelEncoder

import utils_preprocess
import my_callbacks
from global_config import *


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
