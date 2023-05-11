from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D

def get_model(input_shape):
    # input  (624, 129, 1) # input matches with the size of data, which can be obtained as: samples,labels = data.as_numpy_iterator().next() and then samples.shape

    model = Sequential()
    model.add(Conv2D(filters= 32, kernel_size = (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size= (2,2)))

    model.add(Conv2D(filters = 16, kernel_size = (3,3), activation='relu'))    
    model.add(MaxPool2D(pool_size= (2,2)))

    model.add(Flatten())

    model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model 