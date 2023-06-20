# Identification Of Gunshots In Passive Acoustic Monitoring Recordings Using Convolutional Neural Networks
## Bachelor thesis: Maastricht University
## Student: Rosamelia Carioni Porras
## Main supervisor: Pietro Bonizzi 
## Second supervisor: Marijn ten Thij

To get all the required packages do:
pip install -r requirements.txt

The organization of the files and work can be understood as follows:
1. The extraction of samples from the dataframes and 24-hour long recordings provided by the organization can be found under the folder 'pre_processing_data_frames'. 

1.1. In download_audios_training.ipynb the dataframes are read, the time where the event of interest (in this case gunshot) occurs is calculated and added to the data frame. Lastly, the recordings which were not provided by the organisation are dowloaded from amazon web server.  
# TODO: carry on going over this class, everything seems a mess 

2. 


3. Different models architectures were built with hyperparameter tunning, more specifically, Bayesian Optimization. The files where they were built can be found under the jupyter notebooks called: 'build_model_feature.ipynb' where feature = {spectrogram, mfcc_delta, mfcc, spectrogram}. 

4. 

After experiments were perfomed and answer to which pre-processing techniques are best to identify gunshots in environemntal sounds, a final set of 4 different models (with different pre-processing techniques, by varying the arguments in step 7) were trained and tested with the test set provided by the organization. The code for this can be found under the jupyter notebook: 'train_final_models.ipynb'

The results from the models and pre-processing techniques on the test set provided by the organization were obtained on the file: 'results_test_set.ipynb'. 

5. 
Most of the methods used in the training and evaluation of the different models can be found under the 'methods_audio' folder. 
Additionally, to obtain the different models (architectures), the folder 'models' was created. This contains the class 'get_model.py' which returns the model given a specified id. 

6. 

# TODO: 
PRE_PROCESSING

