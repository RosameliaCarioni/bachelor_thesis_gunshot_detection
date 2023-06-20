from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D


def get_model_danaei(input_shape):
    """Given a specific input shape, the method generates a model following the work of Danaei (2021).
    Danaei, D. (2021). Gunshot Detection in Wildlife using Deep Learning.

    Args:
        input_shape

    Returns:
        model
    """

    model = Sequential()
    model.add(Conv2D(filters=40, kernel_size=(3, 3), activation="relu", input_shape=input_shape, strides=(1, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=28, kernel_size=(4, 3), activation="sigmoid", strides=(1, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=24, kernel_size=(4, 3), activation="relu", strides=(1, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(200, activation="sigmoid"))
    model.add(Dense(100, activation="tanh"))
    model.add(Dense(100, activation="tanh"))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def get_model(model_number: int, input_shape=None):
    """Returns a model given an id

    Args:
        model_number (int)
        input_shape: Defaults to None.

    Returns:
        model
    """
    base_path = "/Users/rosameliacarioni/University/Thesis/code/data/models/"
    if model_number == 1:
        model = get_model_danaei(input_shape)
        learning_rate = 0.01
    elif model_number == 2:
        path = base_path + "db_mel_spectrogram"
        model = keras.models.load_model(path)
        learning_rate = 0.002826642869055423
    elif model_number == 3:
        path = base_path + "spectrogram"
        model = keras.models.load_model(path)
        learning_rate = 0.05143203418301916
    elif model_number == 4:
        path = base_path + "mfcc"
        model = keras.models.load_model(path)
        learning_rate = 0.019211177831550998
    elif model_number == 5:
        path = base_path + "mfcc_delta"
        model = keras.models.load_model(path)
        learning_rate = 0.0033434076841967254
    else:
        print("invalid input")

    return model, learning_rate
