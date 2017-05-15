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
INPUT_SHAPE = (1, SR * 10)
