from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import keras_tuner
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from methods_audio import data_handling
from methods_audio import data_augmentation
from methods_audio import denoising 
from methods_audio import model_performance_training
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time

# 1. Get data (file names)
data = data_handling.get_data()

# 2. Read data (transforming file names into waves) <br>
data = data.map(data_handling.read_in_data) 

# 3. Get input for model training 
samples, labels = data_handling.extract_samples_labels(data)

# 4. Split data into train and validation sets
validation_set_size = 0.20
x_train, x_valid, y_train, y_valid = train_test_split(samples, labels, test_size= validation_set_size, random_state=123)

# 5. Transform data to spectograms
type_transformation = 'mfcc'
x_train = data_handling.transform_data(x_train, type_transformation)
x_valid = data_handling.transform_data(x_valid, type_transformation)

## 6. Build model with hyperparameter tunning 

def build_model(hp):
    input = (13, 157, 1)
    model = keras.Sequential()
    
     # Add input layer 
    #matching samples.shape
    model.add(
        Conv2D(
            filters= hp.Int("conv_filters_0", min_value=8, max_value=128, step=16), 
            activation= hp.Choice("conv_activation_0", ["relu", "tanh"]),
            kernel_size = (3,3), 
            input_shape=input
        )
    ) 
    model.add(MaxPool2D(pool_size= (2,2)))

    # Tune the number of Conv layers 
    for i in range(hp.Int("num_conv_layers", 1, 4)):
        model.add(
            Sequential([
                layers.Conv2D(
                    filters=hp.Int(f"conv_filters_{i}", min_value=8, max_value=128, step=16),
                    activation=hp.Choice(f"conv_activation_{i}", ["relu", "tanh"]),
                    kernel_size=(4,3),
                ), 
                layers.MaxPool2D(pool_size=(2,2)),
            ])
        )

    model.add(layers.Flatten())

    # Tune the number of Dense layers and Tune whether to use dropout layer
    for i in range(hp.Int("num_dense_layers", 1, 6)):
            model.add(
                Sequential([
                    layers.Dense(
                        # Tune number of units separately.
                        units=hp.Int(f"dense_units_{i}", min_value=50, max_value=600, step=50),
                        activation=hp.Choice(f"dense_activation_{i}", ["relu", "tanh"]),
                    ), 
                    layers.Dropout(
                        rate=hp.Float(f"dense_dropout_{i}", min_value = 0, max_value = 1)
                    )
                ]) 
            )

    model.add(
        layers.Dense(
        units=1, #because we have 2 classes 
        activation=hp.Choice("activatio_last_layer", ["softmax", "sigmoid"]), 
        )
    )

    # Define the optimizer learning rate as a hyperparameter.
    # sampling="log", the step is multiplied between samples.
    lr= hp.Float("learning_rate", min_value=1e-4, max_value=1e-1, sampling="log")
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr), 
        loss="BinaryCrossentropy", 
        metrics=["accuracy"],
    )
    
    return model

build_model(keras_tuner.HyperParameters())

### Initialize tuner by specifying different arguments 
tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective= "val_accuracy", # we want maximize accuracy 
    max_trials= 50,
    overwrite=True,
    directory="param_optimization",
    project_name="mfcc",
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience=5) 
# patience refers to number of epochs: if the val loss is not improving fter 5 ephocs, we stop it. 
### During the search, the model is called with different hyperparameters 
tuner.search_space_summary()
# Default search space size: number of hyper parameters that we are tunning 
epochs = 50
start_time = time.time()

tuner.search(np.stack(x_train), np.stack(y_train), epochs= epochs, validation_data=(np.stack(x_valid), np.stack(y_valid)), callbacks=[stop_early]) #similar to fit 

end_time = time.time()
elapsed_time = end_time - start_time
print(f"The search took {elapsed_time:.2f} seconds to finish.")

tuner.results_summary()
### After all of that we don't have a model yet but rather a set of hyper parameters. Let's query the results and create a model:
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] 
print(best_hps.values)

learning_rate = best_hps.get('learning_rate')
print(f"Learning rate: {learning_rate}")

model = tuner.hypermodel.build(best_hps)
location = 'data/models/mfcc'

model.save(location)