from audiomentations import AddGaussianNoise, TimeStretch, PitchShift, Reverse, TanhDistortion
import random
import numpy as np 

def time_gaussian_noise(samples, probability):
    """Apply gaussian noise to a sample given a probability

    Args:
        samples
        probability (float): of the augmentation occuring, value between 0 and 1. 

    Returns:
        augmented sample
    """    

    # Similar to Noise Adds
    sample_rate = 8000
    transform = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=probability) 
    augmented_samples = transform(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_reverse(samples, probability):
    """Reverse a sample given a probability

    Args:
        samples
        probability (float): of the augmentation occuring, value between 0 and 1. 

    Returns:
        augmented sample
    """    
    # Similar to rand time shift 
    sample_rate = 8000
    transform = Reverse(p=probability)
    augmented_samples = transform(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_pitch_shift(samples, probability): 
    """Pitch shift the sample up or down by -4 or 4 semitones without changing the tempo

    Args:
        samples
        probability (float): of the augmentation occuring, value between 0 and 1. 

    Returns:
        augmented sample
    """    
    sample_rate = 8000
    transform = PitchShift(min_semitones=-4, max_semitones=4, p=probability)  
    augmented_samples = transform(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_tahn(samples, probability):
    """Pitch shift the sample up or down by -4 or 4 semitones without changing the tempo

    Args:
        samples
        probability (float): of the augmentation occuring, value between 0 and 1. 

    Returns:
        augmented sample
    """    
    sample_rate = 8000
    transform = TanhDistortion( min_distortion=0.01, max_distortion=0.7, p=probability)
    augmented_samples = transform(samples=samples, sample_rate=sample_rate)
    return augmented_samples

def time_strecht(samples, probability): 
    """Time stretches the audio making it faster

    Args:
        samples
        probability (float): of the augmentation occuring, value between 0 and 1. 

    Returns:
        augmented sample
    """    
    sample_rate = 8000
    transform = TimeStretch(min_rate=0.8, max_rate=1.25, p = probability)  
    augmented_samples = transform(samples=samples, sample_rate=sample_rate)
    return augmented_samples


def time_augmentation(samples:list, labels:list, probability:float): 
    """Generate new data from a list of signals (samples), given a probability of occurance
    Args:
        samples (list): list with signals that will be augmented
        labels (list): list with the category of the signal. Either 1 (gunshot) or 0 
        probability (float): of the augmentation occuring, value between 0 and 1. 

    Returns:
        list:  2 lists, one with the original data and the augmented data merged, and the second one with the labels for the data
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

        tahn_sample = time_tahn(sample, probability)
        if not np.all([np.array_equal(arr1, arr2) for arr1, arr2 in zip(sample, tahn_sample)]): 
            # if the samples are not the same, it means that augmentation occured so save the new one 
            new_samples += [tahn_sample]
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

    # Shuffle the lists to reduce any type of bias, ensuring that the paring of signal/label is kept 
    paired_list = list(zip(new_samples, new_labels))
    random.shuffle(paired_list)

    shuffled_signals, suffled_labels = zip(*paired_list) #unziping 

    # Eventhough we unzipped the data, the type is still duple, so when returning it, we cast it to list
    return list(shuffled_signals), list(suffled_labels)
