import numpy as np
import os
from global_config import *


def load_npy_fma(track_id):
    tid_str = '{:06d}'.format(track_id)
    src = np.load(os.path.join(DIR_FMA_NPY, tid_str[:3], tid_str + '.npy'))
    if len(src) <= NSP_SRC:
        src = np.concatenate((np.zeros(NSP_SRC - len(src)), src))
    return src
