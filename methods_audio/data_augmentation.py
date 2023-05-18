from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, TimeMask, Reverse, ClippingDistortion, Gain, Mp3Compression
import random
from methods_audio import data_handling
import numpy as np 

def time_gaussian_noise(samples, probability): 
    # Similar to Noise Adds
    sample_rate = 8000
    augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=probability),
])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_reverse(samples, probability):
    # Similar to rand time shift 
    sample_rate = 8000
    augment = Compose([
    Reverse(p=probability),
    ])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_pitch_shift(samples, probability): 
    # Same as pitch shift, similar to wow resampling 
    sample_rate = 8000
    augment = Compose([  
    PitchShift(min_semitones=-4, max_semitones=4, p=probability),
])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_clip(samples, probability): 
    # Similar to clipping where percentage of points that will be clipped is drawn from a uniform distribution between
    # the two input parameters min_percentile_threshold and max_percentile_threshold.
    sample_rate = 8000
    augment = Compose([  
    ClippingDistortion(min_percentile_threshold= 0, max_percentile_threshold= 40, p= probability),
])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_gain(samples, probability): 
    # Similar to gain
    sample_rate = 8000
    augment = Compose([  
    Gain(min_gain_in_db = 10, max_gain_in_db = 10, p = probability),
])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_compression(samples, probability): 
    # Similar to gain
    sample_rate = 8000
    augment = Compose([  
    Mp3Compression(p = probability),
])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_mask(samples, probability): 
    sample_rate = 8000
    augment = Compose([
    TimeMask(min_band_part=0.0, max_band_part= 0.1, fade = False, p = probability), 
])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_strecht(samples, probability): 
    sample_rate = 8000
    augment = Compose([
    TimeStretch(min_rate=0.8, max_rate=1.25, p = probability),
])  
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    return augmented_samples


def time_augmentation(samples, labels, probability): 
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
        new_samples += [sample]
        new_labels += [label]

        gauss_sample = time_gaussian_noise(sample, probability)
        if not np.all([np.array_equal(arr1, arr2) for arr1, arr2 in zip(sample, gauss_sample)]): 
            # if the samples are not the same, it means that augmentation occured so save the new one 
            new_samples += [gauss_sample]
            new_labels += [label]

        time_sample = time_mask(sample, probability)
        if not np.all([np.array_equal(arr1, arr2) for arr1, arr2 in zip(sample, time_sample)]): 
            # if the samples are not the same, it means that augmentation occured so save the new one 
            new_samples += [time_sample]
            new_labels += [label]

        pitch_sample = time_pitch_shift(sample, probability)
        if not np.all([np.array_equal(arr1, arr2) for arr1, arr2 in zip(sample, pitch_sample)]): 
            # if the samples are not the same, it means that augmentation occured so save the new one 
            new_samples += [pitch_sample]
            new_labels += [label]

        strecht_sample = time_strecht(sample, probability)
        if not np.all([np.array_equal(arr1, arr2) for arr1, arr2 in zip(sample, strecht_sample)]): 
            # if the samples are not the same, it means that augmentation occured so save the new one 
            new_samples += [strecht_sample]
            new_labels += [label]

        reversed_sample = time_reverse(sample, probability)
        if not np.all([np.array_equal(arr1, arr2) for arr1, arr2 in zip(sample, reversed_sample)]): 
            # if the samples are not the same, it means that augmentation occured so save the new one 
            new_samples += [reversed_sample]
            new_labels += [label]

        clip_sample = time_clip(sample, probability)
        if not np.all([np.array_equal(arr1, arr2) for arr1, arr2 in zip(sample, clip_sample)]): 
            # if the samples are not the same, it means that augmentation occured so save the new one 
            new_samples += [clip_sample]
            new_labels += [label]

        gain_sample = time_gain(sample, probability)
        if not np.all([np.array_equal(arr1, arr2) for arr1, arr2 in zip(sample, gain_sample)]): 
            # if the samples are not the same, it means that augmentation occured so save the new one 
            new_samples += [gain_sample]
            new_labels += [label]

        #compressed_sample = time_compression(sample, probability)
        #if not np.all([np.array_equal(arr1, arr2) for arr1, arr2 in zip(sample, compressed_sample)]): 
            # if the samples are not the same, it means that augmentation occured so save the new one 
        #    new_samples += [compressed_sample]
        #    new_labels += [label]

    # Shuffle the lists to reduce any type of bias, ensuring that the paring of signal/label is kept 
    paired_list = list(zip(new_samples, new_labels))
    random.shuffle(paired_list)

    shuffled_signals, suffled_labels = zip(*paired_list) #unziping 

    # Eventhough we unzipped the data, the type is still duple, so when returning it, we cast it to list
    return list(shuffled_signals), list(suffled_labels)
