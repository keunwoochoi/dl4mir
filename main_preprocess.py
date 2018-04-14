""" 16 May 2017, Keunwoo Choi (keunwoo.choi@qmul.ac.uk)

"""
from __future__ import print_function  # (at top of module)

import os
import sys
import pandas as pd
import librosa
import kapre
import numpy as np
import multiprocessing

from global_config import *
import utils_preprocess
import datasets
import pdb


def load_decode_save_fma(track_id):
    """load, decode, and save tracks of FMA.
    Load/Save paths are set by `config.json`.
    track_id : integer. e.g. 2
    """
    tid_str = '{:06d}'.format(track_id)
    audio_path = os.path.join(DIR_FMA_MP3, tid_str[:3], tid_str + '.mp3')
    src, _ = librosa.load(audio_path, sr=SR, duration=LEN_SRC)
    src = src[:NSP_SRC]
    src = src.astype(np.float16)
    try:
        os.mkdir(os.path.join(DIR_FMA_NPY, tid_str[:3]))
    except:
        pass
    np.save(os.path.join(DIR_FMA_NPY, tid_str[:3], tid_str + '.npy'), src)
    print('Done: {}'.format(track_id))



def prep_fma_small():
    """
    Decode fma dataset and store them in numpy arrays.
    Small FMA, 16-bit Float, SR=16000, and 10s, => 2.5GB
    """
    utils_preprocess.make_dir_recursively(DIR_FMA_NPY)

    # pre-process
    tracks = pd.read_csv(os.path.join(DIR_FMA_CSV, 'tracks.csv'), index_col=0, header=[0, 1])
    small = tracks['set', 'subset'] == 'small'
    indices = tracks.loc[small].index

    # decoding
    p = multiprocessing.Pool()
    p.map(load_decode_save_fma, indices)


def prep_jamendo():
    """decode jamendo and ..
    """
    utils_preprocess.make_dir_recursively(DIR_JAMENDO_NPY)
    utils_preprocess.make_dir_recursively(DIR_JAMENDO_DOWNLOAD)
    # pre-process
    duration = None  # TEST
    srcs_sets, ys_sets = datasets.load_jamendo(save_path=DIR_JAMENDO_DOWNLOAD,
                                                duration=duration)
    sets = ['train', 'valid', 'test']
    # decoding audio and label, and save them.
    for set_name, srcs, ys in zip(sets, srcs_sets, ys_sets):
        for i, (src, y) in enumerate(zip(srcs, ys)):
            fn_x = '{}_{:02}_x.npy'.format(set_name, i)
            fn_y = '{}_{:02}_y.npy'.format(set_name, i)
            np.save(os.path.join(DIR_JAMENDO_NPY, fn_x), np.array(src, dtype=np.float16))
            np.save(os.path.join(DIR_JAMENDO_NPY, fn_y), np.array(y, dtype=np.float16))


def main(dataset_name):
    assert dataset_name in ['fma', 'jamendo']
    if dataset_name == 'fma':
        prep_fma_small()
    elif dataset_name == 'jamendo':
        prep_jamendo()


def print_usage():
    print('This script decode audio of some predefined datasets and save in numpy files.')
    print('$ python main_preprocess.py $dataset_name$')
    print('Example:')
    print('$ python main_preprocess.py fma')
    print('$ python main_preprocess.py jamendo')
    print('')
    print('Ps. Make sure you downloaded the dataset and set the dirs/paths in config.json')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage()
    main(sys.argv[1])
