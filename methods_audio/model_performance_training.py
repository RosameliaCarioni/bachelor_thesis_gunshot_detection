from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from methods_audio import data_augmentation
from methods_audio import denoising 
from methods_audio import data_handling
import tensorflow as tf


def train_model(model:tf.keras.Model, x_train:list, y_train:list, x_val:list, y_val:list, batch:int, epoch:int): 
    """Train a model by slicing the data into "batches" of size batch_size, and repeatedly iterating over the entire dataset for a given number of epochs.

    Args:
        model (tf.keras.Model): model to fit
        x_train (list): list of signals to train model
        y_train (list): list of labels for x_train (1: gunshot, 0: no gunshot)
        x_val (list): list of signals to validate model 
        y_val (list): list of labels for x_val (1: gunshot, 0: no gunshot)
        batch (int): number of samples that will be propagated through the network
        epoch (int): how many times mdoel will be trained. In other words, how many times model goes over training set. 

    Returns:
        tuple: model and history of training process
    """        
    
    # if the validation loss doens't improve after 5 epochs, then we stop the training
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience=5) 

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch,
        epochs=epoch,
        validation_data=(x_val, y_val),
        # callbacks=[stop_early],
    )
    return model, history 

def train_performance_k_fold (location_model:str, x:list, y:list, learning_rate:int, epoch:int, batch_size:int, type_augmentation:str, type_denoising:str, low_pass_cutoff:int, low_pass_order:int, type_transformation:str) -> tuple:
    """ 
    https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/ 
    https://repository.tudelft.nl/islandora/object/uuid%3A6f4f3def-f8e0-4820-8b4f-75b0254dadcd 
    https://stackoverflow.com/questions/50997928/typeerror-only-integer-scalar-arrays-can-be-converted-to-a-scalar-index-with-1d
    https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md


    Args:
        location_model (str): _description_
        x (list): _description_
        y (list): _description_
        learning_rate (int): _description_
        epoch (int): _description_
        batch_size (int): _description_

    Returns:
        tuple: _description_
    """    

    # Set values 
    epoch = epoch
    batch = batch_size
    splits = 10

    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=123)
    acc_scores = [] # of test data 
    histories = []
    confusion_matrices = []

    for train, test in kfold.split(x, y):
        # 1. load model 
        model = keras.models.load_model(location_model)
      
        # 2. Compile model ensuring to have wanted metrics
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=learning_rate), 
            loss="BinaryCrossentropy", 
            metrics = ['accuracy', 'Recall', 'Precision'],
        )

        # 3. Split data and transform it from list of numpy.ndarray into numpy.ndarray of numpy.ndarray
        x_train = np.array(x)[train.astype(int)]
        y_train = np.array(y)[train.astype(int)]
        x_valid = np.array(x)[test.astype(int)]
        y_valid = np.array(y)[test.astype(int)]

        # 4. Transform numpy.ndarray into lists
        x_train_list = x_train.tolist() 
        y_train_list = y_train.tolist()
        x_valid_list = x_valid.tolist()
        
        # 5. Data agumentation: if input is none then we do nothing  
        if (type_augmentation == 'signal'):
            x_train_list, y_train_list = data_augmentation.time_augmentation(x_train_list, y_train_list)
        elif(type_augmentation == 'spectrogram'):
            # TODO: work out what to do  
            x_train_list, y_train_list = data_augmentation.spectrogram_augmentaion(x_train_list, y_train_list)
        elif(type_augmentation == 'both'): 
            # TODO: sort this out 
             pass 
    
        # 6. Data denoising: if input is none then we do nothing 
        if (type_denoising == 'spectral'): 
            x_train_list = denoising.apply_spectral(x_train_list)
            x_valid_list =denoising.apply_spectral(x_valid_list)
        elif(type_denoising == 'low_pass'): 
            x_train_list = denoising.apply_low_pass(x_train_list, low_pass_cutoff, low_pass_order)
            x_valid_list =denoising.apply_low_pass(x_valid_list, low_pass_cutoff, low_pass_order)

        # 7. Pad samples so that they all have the same length and transform data to frequency domain
        x_train_list = data_handling.transform_data(x_train_list, type_transformation)
        x_valid_list = data_handling.transform_data(x_valid_list, type_transformation)

        # 8. Transform data from list to np.numpy 
        x_train = np.array(x_train_list) 
        y_train = np.array(y_train_list) 
        x_valid = np.array(x_valid_list) 

        # 8. Save the history of the model for future analysis
        model, hist = train_model(model, x_train, y_train, x_valid, y_valid, batch, epoch)        
        histories.append(hist)
        
        # Display accuracy of validation set 
        # hist.history returns all the metrics. By adding: ['val_accuracy'][epoch-1] we get only the accuracy of the testing set at the last epoch
        print("%s: %.2f%%" % (model.metrics_names[1], hist.history['val_accuracy'][epoch-1] *100)) #TODO: check if it should be epoch-1 or -1 only 
        acc_scores.append(hist.history['val_accuracy'][epoch-1] * 100)

        # Store confusion matrix 
        y_pred = model.predict(x_valid)
        y_pred = [1 if prediction > 0.5 else 0 for prediction in y_pred]
        confusion_mtx = tf.math.confusion_matrix(y_valid, y_pred)
        confusion_matrices.append(confusion_mtx)
    
    print("%.2f%% (+/- %.2f%%)" % (np.mean(acc_scores), np.std(acc_scores)))
    return confusion_matrices, histories

def get_metrics(epoch, histories):

    # Initialize the lists that will be used and returned 
    list_loss = [], list_val_loss = []
    list_precision = [], list_val_precision = []
    list_recall = [], list_val_recall = []
    list_accuracy = [], list_val_accuracy = []

    for i in range(epoch):
        temp_loss = [ hist.history['loss'][i] for hist in histories ]
        list_loss.append(np.mean(temp_loss))
        temp_val_loss = [ hist.history['val_loss'][i] for hist in histories ]
        list_val_loss.append(np.mean(temp_val_loss))

        temp_precision = [ hist.history['precision'][i] for hist in histories ]
        list_precision.append(np.mean(temp_precision))
        temp_val_precision = [ hist.history['val_precision'][i] for hist in histories ]
        list_val_precision.append(np.mean(temp_val_precision))

        temp_recall = [ hist.history['recall'][i] for hist in histories ]
        list_recall.append(np.mean(temp_recall))
        temp_val_recall = [ hist.history['val_recall'][i] for hist in histories ]
        list_val_recall.append(np.mean(temp_val_recall))

        temp_accuracy = [ hist.history['accuracy'][i] for hist in histories ]
        list_accuracy.append(np.mean(temp_accuracy))
        temp_val_accuracy = [ hist.history['val_accuracy'][i] for hist in histories ]
        list_val_accuracy.append(np.mean(temp_val_accuracy))
    return list_loss, list_val_loss, list_precision, list_val_precision, list_recall, list_val_recall, list_accuracy, list_val_accuracy


def plot_performance(list_train, list_validation, title_plot:str, title_y_label:str, destination_file:str): 
    """_summary_

    Args:
        list_train (_type_): can be loss, precision, recall, accuracy
        list_validation (_type_): _description_
        title_plot (str): _description_
        title_y_label (str): _description_
        destination_file (str): '...png'
    """    
    plt.title(title_plot)
    plt.plot(list_train, 'r')
    plt.plot(list_validation, 'b')
    plt.xlabel('Epoch')
    plt.ylabel(title_y_label)
    plt.legend(['Train', 'Test'])
    plt.grid()
    plt.savefig(destination_file)
    plt.close() # this stops it from showing 
    #plt.show() # this shows it 

def plot_confusion_matrix (confusion_matrices, destination_file:str, title_plot:str): 
    """_summary_
    https://www.tensorflow.org/tutorials/audio/simple_audio

    Args:
        confusion_matrices (_type_): 
        destination_file (str): '...png'
        title_plot (str):
    """    

    mean_matrix = np.mean(confusion_matrices, axis=0)

    label_names = ['Gunshot' ,'No gunshot']
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_matrix,
                xticklabels=label_names,
                yticklabels=label_names,
                annot=True, fmt='g')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title_plot)
    plt.savefig(destination_file)
    plt.close() # this stops it from showing 
    # plt.show() # this shows it 


# evaluate the model on a validation set
#loss, accuracy = model.evaluate(np.stack(x_valid), np.stack(y_valid))

# print the evaluation results
#print(f'Validation loss: {loss:.4f}')
#print(f'Validation accuracy: {accuracy:.4f}')