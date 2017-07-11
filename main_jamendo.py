""" 16 May 2017, Keunwoo Choi (keunwoo.choi@qmul.ac.uk)

It assumes FMA-small dataset is downloaded and pre-processed by main_preprocess.py.

"""
from __future__ import print_function  # (at top of module)

import os
import pandas as pd
import numpy as np
import models_frame
import models_MLP

from sklearn.preprocessing import LabelEncoder

import utils_preprocess
import my_callbacks
from global_config import *

import pdb

# TODO: add MLP models.

N_DATA = {'train': 61,
          'valid': 16,
          'test': 16}

N_HOP = 256


def load_all_data(set_name):
    srcs = []
    ys = []
    for i in range(N_DATA[set_name]):
        srcs.append(np.load(os.path.join(DIR_JAMENDO_NPY, '{}_{:02}_x.npy'.format(set_name, i))))
        ys.append(np.load(os.path.join(DIR_JAMENDO_NPY, '{}_{:02}_y.npy'.format(set_name, i))))
    return srcs, ys


def y_sample_to_frame(y):
    """
    y : (batch_size, N) --> (batch_size, M), M = number of frames
    """
    n_hop = N_HOP
    nsp_y = len(y)
    ret = np.array([np.round(np.mean(y[max(0, (i - 1) * n_hop): min(nsp_y, (i + 1) * n_hop)])) \
                    for i in xrange(nsp_y // n_hop)], dtype=np.int)
    return ret


def data_gen(set_name, nsp_input, is_shuffle, is_infinite, batch_size):
    """Data generator for Jamendo

    Mostly, ['train', N, True, True, K1],
            ['valid', N, False, False, K2],
            ['test', None, False, False, 1].

    set_name: string, 'train', 'valid', 'test'
    nsp_input: num_sample of input (length) [sample]. if None, it uses the whole length.
    is_shuffle: boolean, whether shuffle every batch if True.
    is_infinite; boolean, whether the generator is infinite or not
    batch_size: integer, size of batch. len(df_subset) % batch_size should be 0.
    """
    n_data = N_DATA[set_name]
    assert batch_size <= n_data
    n_batch = n_data // batch_size

    if n_data % batch_size != 0:
        if not is_shuffle:
            print('WARNING: The generator for {} will ignore the last {} samples'.format(set_name, n_data % batch_size))

    srcs, ys = load_all_data(set_name)
    # ys = [y_sample_to_frame(y) for y in ys]

    while True:
        for batch_i in xrange(n_batch):
            if is_shuffle:
                data_idxs = np.random.choice(n_data, batch_size, replace=False)
            else:
                data_idxs = range(batch_i * batch_size, (batch_i + 1) * batch_size)

            src_batch = []
            y_batch = []
            for data_i in data_idxs:
                src = srcs[data_i]  # (1, N)
                y = ys[data_i]
                nsp_src = src.shape[1]
                if nsp_input is None:
                    src = src[:, : (nsp_src // N_HOP) * N_HOP]
                    y = y[: (nsp_src // N_HOP) * N_HOP]
                else:
                    offset = np.random.randint(low=0, high=nsp_src - nsp_input)
                    src = src[:, offset: offset + nsp_input]
                    y = y[offset: offset + nsp_input]

                src_batch.append(src)
                y_batch.append(y)

            src_batch = np.array(src_batch, dtype=K.floatx())

            y_batch = [y_sample_to_frame(y) for y in y_batch]
            y_batch = np.array(y_batch, dtype=np.int)  # (batch_size, N)
            y_batch = np.eye(2, dtype=K.floatx())[y_batch]
            yield src_batch, y_batch

        if not is_infinite:
            break


def main(model_name, exp_name='fma'):
    """
    DO it!
    """
    assert model_name in ['crnn', 'lstm', 'lstm_bi']
    print("-" * 60)
    print("Keunwoo: Welcome! Lets do something deep with Jamendo dataset.")
    print("         I'm assuming you finished pre-processing.")
    print("         We're gonna use {} model".format(model_name))

    print("It's a good practice to use callbacks in Keras.")
    callbacks = my_callbacks.get_callbacks(name=exp_name)
    early_stopper, model_saver, weight_saver, csv_logger = callbacks  # just to show you

    print("Preparing data generators for training and validation...")
    train_batch_size = 40
    valid_batch_size = 5
    steps_per_epoch = 30
    gen_train = data_gen('train', SR * 10, is_shuffle=True, is_infinite=True, batch_size=train_batch_size)
    gen_valid = data_gen('valid', SR * 10, is_shuffle=False, is_infinite=True, batch_size=valid_batch_size)

    print("Keunwoo: Getting model...")
    if model_name == 'crnn':
        model = models_frame.model_convrnn(n_out=2)
    elif model_name == 'lstm':
        model = models_frame.model_lstm_leglaive_icassp2014(n_out=2, bidirectional=False)
    elif model_name == 'lstm_bi'
        model = models_frame.model_lstm_leglaive_icassp2014(n_out=2, bidirectional=True)

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    print("Keunwoo: Starting to train...")
    model.fit_generator(gen_train, steps_per_epoch, epochs=30,
                        callbacks=callbacks,
                        validation_data=gen_valid,
                        validation_steps=N_DATA['valid'] // valid_batch_size)

    print("Keunwoo: Training is done. Loading the best weights...")
    model.load_weights("{}_best_weights.h5".format(exp_name))
    gen_test = data_gen('test', None, is_shuffle=False, is_infinite=True, batch_size=1)

    print("         Evaluating...")
    scores = model.evaluate_generator(gen_test, N_DATA['test'])

    print('Keunwoo: Done for {}!'.format(model_name))
    print("         test set loss:{}".format(scores[0]))
    print("         test set accuracy: {}%".format([scores[1]]))


if __name__ == '__main__':
    main('crnn', 'jamendo_crnn')
    main('lstm', 'jamendo_lstm')
    main('lstm_bi', 'jamendo_lstm_bi')
