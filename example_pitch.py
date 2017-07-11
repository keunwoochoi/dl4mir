import numpy as np
import librosa
import keras
from matplotlib import pyplot as plt


def sin_wave(secs, freq, sr, gain):
    '''
    Generates a sine wave of frequency given by freq, with duration of secs.
    '''
    t = np.arange(sr * secs)
    return gain * np.sin(2 * np.pi * freq * t / sr)


def whitenoise(gain, shape):
    '''
    Generates white noise of duration given by secs
    '''
    return gain * np.random.uniform(-1., 1., shape)


class DataGen:
    def __init__(self, sr=16000, batch_size=32):
        np.random.seed(1209)
        self.pitches = [440., 466.2, 493.8, 523.3, 554.4, 587.3,
                        622.3, 659.3, 698.5, 740., 784.0, 830.6]

        self.sr = sr
        self.n_class = len(self.pitches)  # 12 pitches
        self.secs = 1.
        self.batch_size = batch_size
        self.sins = []
        self.labels = np.eye(self.n_class)[range(0, self.n_class)]  # 1-hot-vectors

        for freq in self.pitches:
            cqt = librosa.cqt(sin_wave(self.secs, freq, self.sr, gain=0.5), sr=sr,
                              fmin=220, n_bins=36)[:, 1]  # use only one frame!

            self.sins.append(cqt)

        self.cqt_shape = cqt.shape  # (36, )

    def next(self):
        choice = np.random.choice(12, size=self.batch_size,
                                  replace=True)
        noise_gain = 0.1 * np.random.random_sample(1)
        noise = whitenoise(noise_gain, self.cqt_shape)
        xs = [noise + self.sins[i] for i in choice]
        ys = [self.labels[i] for i in choice]

        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def main():
    """
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    dense_1 (Dense)              (None, 12)                432
    _________________________________________________________________
    activation_1 (Activation)    (None, 12)                0
    =================================================================
    Total params: 432
    Trainable params: 432
    Non-trainable params: 0
    _________________________________________________________________
    """
datagen = DataGen()
print(datagen.cqt_shape)
print(datagen.n_class)

model = keras.models.Sequential()
model.add(keras.layers.Dense(datagen.n_class, use_bias=False,
                             input_shape=datagen.cqt_shape))
model.add(keras.layers.Activation('softmax'))

model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9,
                                             decay=1e-6, nesterov=True),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


model.fit_generator(datagen, steps_per_epoch=100, epochs=10)

loss = model.evaluate_generator(datagen, steps=10)
    print loss
    weights = model.get_weights()[0] # (36, 12)

    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    main()
