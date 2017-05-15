import kapre
import os

import models_excerpt

PATH_DOWNLOAD = '../dataset'


def main():
    # download FMA dataset (small)
    try:
        os.mkdir(PATH_DOWNLOAD)
    except:
        pass
    kapre.datasets.load_fma(save_path=PATH_DOWNLOAD, size='small')

    # training
    model = models_excerpt.model_conv3x3_ismir2016_choi(n_out=8)


if __name__ == '__main__':
    main()
