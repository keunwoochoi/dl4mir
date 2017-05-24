import keras


def get_callbacks(name):
    early_stopper = keras.callbacks.EarlyStopping(patience=5)
    model_saver = keras.callbacks.ModelCheckpoint("{}_best_model.h5".format(name),
                                                  save_best_only=True)
    weight_saver = keras.callbacks.ModelCheckpoint("{}_best_weights.h5".format(name),
                                                   save_best_only=True,
                                                   save_weights_only=True)
    csv_logger = keras.callbacks.CSVLogger("{}.log".format(name))
    return [early_stopper, model_saver, weight_saver, csv_logger]
