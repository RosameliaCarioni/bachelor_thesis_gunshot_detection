from scipy.signal import butter, lfilter
import numpy as np
import scipy.io.wavfile as wavfile
import noisereduce as nr


# This function saves clips
def save_denoised(reduced_noise, rate, destination_file):
    """Save a signal
        https://stackoverflow.com/questions/64813162/read-wav-file-with-tf-audio-decode-wav
    Args:
        reduced_noise: signal to save
        rate: sample rate
        destination_file: path to where to save sample
    """
    # because the denoised clips will be used by tf.audio.decode_wav and this only takes 16-bit files, the denoised audios are saved as int16
    wavfile.write(destination_file, rate, reduced_noise.astype(np.int16))


# SPECTRAL GATING METHODS


def spectral(wave):
    """Apply spectral gating denoising.
        Allows a main signal to pass through only when it is above a set threshold: the gate is "open".
        Works by computing a spectrogram of a signal and estimating a noise threshold (or gate) for each frequency band of that signal/noise.
        That threshold is used to compute a mask, which gates noise below the frequency-varying threshold.
        Stationary Noise Reduction: Keeps the estimated noise threshold at the same level across the whole signal
        https://pypi.org/project/noisereduce/

    Args:
        wave: signal to denoise

    Returns:
        denoised signal
    """
    rate = 8000
    reduced_noise = nr.reduce_noise(y=wave, sr=rate, stationary=True, win_length=256, hop_length=128)
    return reduced_noise


def apply_spectral(signals, differentiation=False):
    """Denoise a list of signals by applying spectral gating denoising

    Args:
        signals (list): signals that will be denoised
        differentiation (bool, optional): _description_. Defaults to False.

    Returns:
        list: with denoised signals
    """

    denoised_signals = []
    if differentiation == True:
        for wave in signals:
            wave = np.diff(wave)
            denoised_wave = spectral(wave)
            denoised_signals.append(denoised_wave)
    else:
        for wave in signals:
            denoised_wave = spectral(wave)
            denoised_signals.append(denoised_wave)

    return denoised_signals


# LOW PASS FILTER METHODS
def low_pass(wave, cutoff, order=4):
    """Apply low pass filter to a sample.

    Args:
        wave: signal to denoise
        cutoff: only frequencies bellow this threshold would be kept. Value is in Hz.
        order (int, optional): defaults to 4.

    Returns:
        denoised signal
    """
    sample_rate = 8000
    b, a = butter(order, cutoff, fs=sample_rate, btype="low", analog=False)
    filtered_data = lfilter(b, a, wave)
    return filtered_data


def apply_low_pass(signals, cutoff, differentiation=False, order=4):
    """Denoise a list of signals by applying spectral gate denoising

    Args:
        signals (list): signals that will be denoised
        cutoff (int):  only frequencies bellow this threshold would be kept. Value is in Hz.
        differentiation (bool, optional): defaults to False
        order (int, optional): defaults to 4

    Returns:
        list: denoised signals
    """
    denoised_signals = []
    if differentiation == True:
        for wave in signals:
            wave = np.diff(wave)
            denoised_wave = low_pass(wave, cutoff, order)
            denoised_signals.append(denoised_wave)
    else:
        for wave in signals:
            denoised_wave = low_pass(wave, cutoff, order)
            denoised_signals.append(denoised_wave)

    return denoised_signals


def differentiate_signal(file_name):
    """Differentiate a signal by substracting
    https://dsp.stackexchange.com/questions/56604/bandpass-filter-for-audio-wav-file
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

    Args:
        file_name: path of signal

    Returns:
        Differentiated signal and rate
    """
    rate, data = wavfile.read(file_name)
    data = data - data.mean()  # first center the signal
    return np.diff(data), rate
