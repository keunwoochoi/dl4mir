""" 16 May 2017, Keunwoo Choi (keunwoo.choi@qmul.ac.uk)

"""
from __future__ import print_function  # (at top of module)

import os
import sys
import pandas as pd
import librosa
import numpy as np
import multiprocessing

from global_config import *


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
    # recursively create folder to save
    dirs = DIR_FMA_NPY.split('/')
    npy_path_sub = ''
    for dr in dirs:
        npy_path_sub = os.path.join(npy_path_sub, dr)
        try:
            os.mkdir(npy_path_sub)
        except:
            pass

    # pre-process
    tracks = pd.read_csv(os.path.join(DIR_FMA_CSV, 'tracks.csv'), index_col=0, header=[0, 1])
    small = tracks['set', 'subset'] == 'small'
    indices = tracks.loc[small].index

    # decoding
    p = multiprocessing.Pool()
    p.map(load_decode_save_fma, indices)


def main(dataset_name):
    if dataset_name == 'fma':
        prep_fma_small()


def print_usage():
    print('This script decode audio of some predefined datasets and save in numpy files.')
    print('$ python main_preprocess.py $dataset_name$')
    print('Example:')
    print('$ python main_preprocess.py fma')
    print('')
    print('Ps. Make sure you downloaded the dataset and set the dirs/paths in config.json')


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except:
        print_usage()
