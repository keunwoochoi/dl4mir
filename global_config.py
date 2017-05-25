import json
from keras import backend as K

if K.image_data_format() == 'channels_first':
    print('Channel-first, i.e., (None, n_ch, n_freq, n_time)')
    channel_axis = 1
    freq_axis = 2
    time_axis = 3
else:
    print('Channel-last, i.e., (None, n_freq, n_time, n_ch)')
    channel_axis = 3
    freq_axis = 1
    time_axis = 2

SR = 16000  # TODO: 8000? (only for speed)
LEN_SRC = 10.
NSP_SRC = int(SR * LEN_SRC)
INPUT_SHAPE = (1, NSP_SRC)

with open('config.json') as json_data:
    config = json.load(json_data)

DIR_FMA_MP3 = config['dir_fma_mp3']
DIR_FMA_CSV = config['dir_fma_csv']
DIR_FMA_NPY = config['dir_fma_npy']

DIR_JAMENDO_DOWNLOAD = config['dir_jamendo_download']
DIR_JAMENDO_NPY = config['dir_jamendo_npy']
