from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, TimeMask, SpecCompose, SpecFrequencyMask
import random


def time_gaussian_noise(samples): 
    sample_rate = 8000
    augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_mask(samples): 
    sample_rate = 8000
    augment = Compose([
    TimeMask(min_band_part=0.0, max_band_part= 0.1, fade = False, p = 0.5), 
])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_pitch_shift(samples): 
    sample_rate = 8000
    augment = Compose([  
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_strecht(samples): 
    sample_rate = 8000
    augment = Compose([
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples


def spectrogram_frequency_mask (spectrogram):
    
    augment = SpecCompose([
        SpecFrequencyMask(p=0.5),
    ])
    augmented_spectrogram = augment(spectrogram)

    return augmented_spectrogram

def time_augmentation(samples, labels): 
    """
        This method generates new data from signals 

        :param samples: list with signals that will be augmented
        :type samples: list of numpy.ndrray
        :param labels: list with the category of the signal. Either 1 (gunshot) or 0 
        :type arg2: list of int
        :return: 2 new lists with the original data and the augmented data

    """
    
    new_samples = []
    new_labels = []
    for sample, label in zip(samples, labels): 
        gauss_sample = time_gaussian_noise(sample)
        time_sample = time_mask(sample)
        pitch_sample = time_pitch_shift(sample)
        strecht_sample = time_strecht(sample)
        new_samples += [sample,gauss_sample,time_sample, pitch_sample,  strecht_sample]
        new_labels += [label]*5

    # Shuffle the lists to reduce any type of bias, ensuring that the paring of signal/label is kept 
    paired_list = list(zip(new_samples, new_labels))
    random.shuffle(paired_list)

    shuffled_signals, suffled_labels = zip(*paired_list) #unziping 

    return shuffled_signals, suffled_labels

def spectrogram_augmentaion(spectrogram, labels): 
    """
        This method generates new data from spectrograms 

        :param spectrogram: list with spectrograms that will be augmented
        :type samples: list of numpy.ndrray
        :param labels: list with the category of the signal. Either 1 (gunshot) or 0 
        :type arg2: list of int
        :return: 2 new lists with the original data and the augmented data
    """
    new_spects = []
    new_labels = []
    for spect, label in zip(spectrogram, labels): 
        spect_augmented = spectrogram_frequency_mask(spect)
        new_spects += [spect, spect_augmented]
        new_labels += [label]*2

    # Shuffle the lists to reduce any type of bias, ensuring that the paring of signal/label is kept 
    paired_list = list(zip(new_spects, new_labels))
    random.shuffle(paired_list)

    shuffled_spect, suffled_labels = zip(*paired_list) #unziping 

    return shuffled_spect, suffled_labels