from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D


def get_model(input_shape):
    # Obtained from: Danaei, D. (2021). Gunshot Detection in Wildlife using Deep Learning. 

    model = Sequential()
    model.add(Conv2D(filters= 40, kernel_size = (3,3), activation='relu', input_shape=input_shape, strides = (1,1)))
    model.add(MaxPool2D(pool_size= (2,2), strides = (2,2)))

    model.add(Conv2D(filters = 28, kernel_size = (4,3), activation='sigmoid', strides = (1,1)))    
    model.add(MaxPool2D(pool_size= (2,2), strides = (2,2)))

    model.add(Conv2D(filters = 24, kernel_size = (4,3), activation='relu', strides = (1,1)))    
    model.add(MaxPool2D(pool_size= (2,2), strides = (2,2)))

    model.add(Flatten())

    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model 

# to see model: 
# model.summary()