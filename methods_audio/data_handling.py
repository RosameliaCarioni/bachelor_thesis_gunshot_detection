import tensorflow as tf
import tensorflow_io as tfio
import os 
import librosa
import numpy as np

# TODO: delete this method, it was only used for debugging 
def get_data2():
    """
    https://www.youtube.com/watch?v=ZLIPkmmDJAc&t=1468s&ab_channel=NicholasRenotte 

    Returns:
        All data including gunshots and no gunshots, with the form (file_path,label)
    """    

    # Data from the elephant listening project 
    general_path = os.path.join('/Users','rosameliacarioni','University','Thesis','code','data', 'train', 'ollie')

    
    gunshot_files = [os.path.join(general_path, 'pnnn*'), os.path.join(general_path, 'ecoguns*')]

    gunshot = tf.data.Dataset.list_files(gunshot_files, shuffle=False)  # setting shuffle to False so that the files get always read in the same order 

    #to see how many files are in each group: 
    #num_elements = tf.data.experimental.cardinality(no_gunshot).numpy()

    # Add labels to the data 
    gunshot = tf.data.Dataset.zip((gunshot, tf.data.Dataset.from_tensor_slices(tf.ones(len(gunshot)))))

    # Concatenate gunshots and no gunshots and shuffle data 
    data = gunshot
    data = data.cache()
    data = data.shuffle(buffer_size=1000, seed = 123) # mixing training samples 1000 at the time  

    return data

def get_data():
    """
    https://www.youtube.com/watch?v=ZLIPkmmDJAc&t=1468s&ab_channel=NicholasRenotte 

    Returns:
        All data including gunshots and no gunshots, with the form (file_path,label)
    """    

    # Data from the elephant listening project 
    general_path = os.path.join('/Users','rosameliacarioni','University','Thesis','code','data', 'train', 'Clips_ELP')

    # To ensure that both classes have same of samples and to increase the number of gunshots, 
    # I extracted extra data from: https://data.mendeley.com/datasets/x48cwz364j/3 
    background_path = os.path.join('/Users','rosameliacarioni','University','Thesis','code','data', 'train', 'Clips_Mendeley_no_gunshot')
    guns_path = os.path.join('/Users','rosameliacarioni','University','Thesis','code','data', 'train', 'Clips_Mendeley_gunshot')

    gunshot_files = [os.path.join(general_path, 'pnnn*'), os.path.join(general_path, 'ecoguns*'), os.path.join(guns_path, '*\.wav')]

    no_gunshot_files = [os.path.join(general_path, 'other*'), os.path.join(background_path, '*\.wav')] 
    gunshot = tf.data.Dataset.list_files(gunshot_files, shuffle=False)  # setting shuffle to False so that the files get always read in the same order 
    no_gunshot = tf.data.Dataset.list_files(no_gunshot_files, shuffle=False) # setting shuffle to False so that the files get always read in the same order 

    #to see how many files are in each group: 
    #num_elements = tf.data.experimental.cardinality(no_gunshot).numpy()

    # Add labels to the data 
    gunshot = tf.data.Dataset.zip((gunshot, tf.data.Dataset.from_tensor_slices(tf.ones(len(gunshot)))))
    no_gunshot= tf.data.Dataset.zip((no_gunshot, tf.data.Dataset.from_tensor_slices(tf.zeros(len(no_gunshot)))))

    # Concatenate gunshots and no gunshots and shuffle data 
    data = gunshot.concatenate(no_gunshot)
    data = data.cache()
    data = data.shuffle(buffer_size=1000, seed = 123) # mixing training samples 1000 at the time  

    return data


def read_in_data(file_name, label): 
    """_summary_

    Args:
        file_name (_type_): A tensor of dtype "string", with the file contents.
        label (_type_): A tensor of dtype "numpy.float32", with the classification of the file content: 1 (gunshot) or 0 (no gunshot)

    Returns:

    """    

    file_contents = tf.io.read_file(file_name) #retuns a string 
    wave, _ = tf.audio.decode_wav(file_contents, desired_channels=1) # transforms string into actual wav
    wave = wave - tf.reduce_mean(wave) # remove the mean 
    wave = wave / tf.reduce_max(tf.abs(wave)) #normalize 
    wave = tf.squeeze(wave, axis= -1) #removes axis 
    return wave, label


def extract_samples_labels(data): 
    iterator = data.as_numpy_iterator()
    x = []
    y = []
    while True:
        try: 
            x_temp, y_temp = iterator.next()
            x.append(x_temp)
            y.append(y_temp)
        except Exception:
            break 
    return x,y 

def pad_sample(wave):
    # Define length of sample 
    size_samples = 10 # value in seconds 
    sample_rate = 8000 
    max_lenght = size_samples * sample_rate 

    # Padding with 0s
    wave = wave[:max_lenght] #grab first elements up to max(lengths)
    zero_padding = tf.zeros(max_lenght - tf.shape(wave), dtype=tf.float32) # pad with zeros what doesn't meet full length 
    wave = tf.concat([zero_padding, wave],0) 
    return wave 

def convert_to_spectrogram(wave):
    # 1. Fast fourier transform 
    spectrogram = tf.signal.stft(wave, frame_length=256, frame_step=128)  # Paper: 'Automated detection of gunshots in tropical forests using CNN' 
    # frame_length =  window length in samples --- frame_step = number of samples to step
    # 'Time frequency compromise': if window size is small: you get good time resolution in exchange of poor frequency resolution 

    # 2. Obtain the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)

    # 3. Tranform it into appropiate format for deep learning model by adding the channel dimension (in this case 1)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

def convert_to_mel_spectrogram(wave):
    sr_audio = 8000
    number_mels_filterbanks = 128 
    # 1. Fast fourier transform 
    spectrogram = tf.signal.stft(wave, frame_length=256, frame_step=128)  # Paper: 'Automated detection of gunshots in tropical forests using CNN' 
    # 2. Obtain the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)
    # 3. Convert to mel-spectrogram 
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=sr_audio, mels=number_mels_filterbanks, fmin=0, fmax=4000) 

    # 4. Tranform it into appropiate format for deep learning model by adding the channel dimension
    mel_spectrogram = tf.expand_dims(mel_spectrogram, axis=2)
    return mel_spectrogram

def convert_to_mel_spectrogram_db_scale(wave):
    sr_audio = 8000
    number_mels_filterbanks = 128 
    # 1. Fast fourier transform 
    spectrogram = tf.signal.stft(wave, frame_length=256, frame_step=128)  # Paper: 'Automated detection of gunshots in tropical forests using CNN' 
    # 2. Obtain the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)

    # 3. Convert into mel-spectrogram
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=sr_audio, mels=number_mels_filterbanks, fmin=0, fmax=4000) #TODO: play with this numbers 

    # 4. Convert the mel-spectrogram into db scale
    db_mel_spectrogram = librosa.power_to_db(mel_spectrogram.numpy())
    dbscale_mel_spectrogram = tf.convert_to_tensor(db_mel_spectrogram)

    # 5. Tranform it into appropiate format for deep learning model by adding the channel dimension
    dbscale_mel_spectrogram = tf.expand_dims(dbscale_mel_spectrogram, axis=2)
    return dbscale_mel_spectrogram

# TODO: not  sure if this is going to be used, if it is then it needs to be added to the method that handles the type of feature 
def convert_to_mel_spectrogram_log_db(wave):
    sr_audio = 8000
    number_mels_filterbanks = 128 
    # 1. Fast fourier transform 
    spectrogram = tf.signal.stft(wave, frame_length=256, frame_step=128)  # Paper: 'Automated detection of gunshots in tropical forests using CNN' 
    # 2. Obtain the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)

    # 3. Convert into mel-spectrogram
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=sr_audio, mels=number_mels_filterbanks, fmin=0, fmax=4000) #TODO: play with this numbers 

    # 4. Convert the mel-spectrogram into db scale
    db_mel_spectrogram = librosa.power_to_db(mel_spectrogram.numpy())

    # 5. Convert mel-spectrogram in db scale to log-mel spectrogram 
    log_mel_spectrogram = tf.math.log(db_mel_spectrogram + 1e-6) # the + 1e-6 is to ensure not taking the log of 0 
    

    log_mel_spectrogram = tf.convert_to_tensor(log_mel_spectrogram)

    # 5. Tranform it into appropiate format for deep learning model by adding the channel dimension
    dbscale_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, axis=2)
    return dbscale_mel_spectrogram

def convert_to_mfcc(wave): 
    # https://www.youtube.com/watch?v=WJI-17MNpdE&t=575s&ab_channel=ValerioVelardo-TheSoundofAI
    sr = 8000 
    mfccs = librosa.feature.mfcc(wave.numpy(), n_mfcc = 13, sr = sr)
    mfccs = tf.convert_to_tensor(mfccs) 
    mfccs = tf.expand_dims(mfccs, axis=2)

    return mfccs 

def convert_to_mfcc_and_delta(wave): 
    # https://www.youtube.com/watch?v=WJI-17MNpdE&t=575s&ab_channel=ValerioVelardo-TheSoundofAI
    sr = 8000 
    mfccs = librosa.feature.mfcc(wave.numpy(), n_mfcc = 13, sr = sr)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    result = np.concatenate((mfccs, delta_mfccs, delta2_mfccs)) 
    mfccs = tf.convert_to_tensor(result) 
    mfccs = tf.expand_dims(mfccs, axis=2)

    return mfccs 

def transform_data (waves, type_transformation):
    transformed_signals = []

    if (type_transformation == 'spectrogram'):
        for wave in waves: 
            wave = pad_sample(wave)
            transformed_wave = convert_to_spectrogram(wave)
            transformed_signals.append(transformed_wave)

    elif (type_transformation == 'mel_spectrogram'):
       for wave in waves: 
            wave = pad_sample(wave)
            transformed_wave = convert_to_mel_spectrogram(wave)
            transformed_signals.append(transformed_wave)

    elif (type_transformation == 'db_mel_spectrogram'): 
        for wave in waves: 
            wave = pad_sample(wave)
            transformed_wave = convert_to_mel_spectrogram_db_scale(wave)
            transformed_signals.append(transformed_wave)

    elif (type_transformation == 'mfcc'): 
        for wave in waves: 
            wave = pad_sample(wave)
            transformed_wave = convert_to_mfcc(wave)
            transformed_signals.append(transformed_wave)

    elif (type_transformation == 'mfcc_delta'): 
        for wave in waves: 
            wave = pad_sample(wave)
            transformed_wave = convert_to_mfcc_and_delta(wave)
            transformed_signals.append(transformed_wave)
            
    return transformed_signals


# Function pads the clips such that they all have the same length and it converts them into the specified 'type'. 
def preprocess(file_path, label, type): 
    """This function pads all clips so that their length is 10 seconds and it transforms them to eithter: spectrogram, mel spectrogram or 
    mel spectrogram in db scale. 

    Args:
        file_path (string)
        label (int): 1 or 0, depending if it's a gunshot or not. 
        type (string): spectrogram, mel_spectrogram, mel_spectrogram_db

    Returns:
        int: The sum of the two numbers. #TODO 
    """

    # Load data
    wave, sr = load_data(file_path)
    max_lenght = 80000 # = 10* 8000, this means 10 seconds 

    # Padding 
    wave = wave[:max_lenght] #grab first elements up to max(lengths)
    zero_padding = tf.zeros(max_lenght - tf.shape(wave), dtype=tf.float32) # pad with zeros what doesn't meet full length 
    wave = tf.concat([zero_padding, wave],0) 

    # Transform data into specified 'type'
    if type == 'spectrogram':
        new_data = convert_to_spectrogram(wave)
    elif type == 'mel_spectrogram': 
        new_data = convert_to_mel_spectrogram(wave)
    elif type == 'mel_spectrogram_db': 
        new_data = convert_to_mel_spectrogram_db_scale(wave)

    return new_data, label

