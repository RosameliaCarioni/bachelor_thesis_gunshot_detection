import pandas as pd
import wave
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile


def joint_plots(origen_file: str, destination_file: str):
    """Generate plot with 3 subplots in: time-domain representation, magnitude spectrum and spectrogram of an audio sample

    Args:
        origen_file (str)
        destination_file (str): to save the plots
    """
    # Load audio file and apply FFT
    sample_rate, samples = wavfile.read(origen_file)

    samples = samples - samples.mean()  # remove the mean

    fft_values = np.fft.fft(samples)
    fig = plt.figure(figsize=(10, 12))

    # Plot time-domain representation of the audio signal
    ax1 = plt.subplot(311)
    t = np.arange(len(samples)) / float(sample_rate)
    ax1.plot(t, samples)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time-domain representation")
    ax1.grid()

    # Plot frequency-domain representation of the audio signal: how much each frequency is contributing to overall sound of audio file
    ax2 = plt.subplot(312)
    ax2.plot(abs(fft_values))
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Magnitude spectrum plot: Contribution of each frequency")
    ax2.grid()

    # Plot spectrogram of the audio signal
    ax3 = plt.subplot(313)
    t_audio = len(samples) / sample_rate

    ax3.specgram(samples, Fs=sample_rate, vmin=-20, vmax=50)
    ax3.set_title("Spectogram")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_xlabel("Time (s)")
    ax3.set_xlim(0, t_audio)

    # Combine all the plots into a single figure
    plt.tight_layout()
    # plt.savefig(destination_file)
    plt.show()
    plt.close()
    plt.show()


def plot_freq_spectrum(origen_file: str, destination_file: str):
    """Generate spectrogram

    Args:
        origen_file (str)
        destination_file (str)

    """

    wave_object = wave.open(origen_file, "rb")
    channel = wave_object.getnchannels()
    n_samples = wave_object.getnframes()
    sample_freq = wave_object.getframerate()
    t_audio = n_samples / sample_freq
    signal_wave = wave_object.readframes(n_samples)
    if channel != 1:
        raise Exception("Some sample have more than one channel")
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    plt.figure(figsize=(19, 5))
    plt.specgram(signal_array, Fs=sample_freq, vmin=-20, vmax=50)
    plt.title("Spectogram")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.xlim(0, t_audio)
    plt.colorbar()
    plt.savefig(destination_file)
    plt.close()
    # plt.show()


def plot_time(origen_file: str, destination_file: str):
    """ "Generate plot with time-domain representation

    Args:
        origen_file (str):
        destination_file (str)

    """
    wave_object = wave.open(origen_file, "rb")
    channel = wave_object.getnchannels()
    n_samples = wave_object.getnframes()
    sample_freq = wave_object.getframerate()
    t_audio = n_samples / sample_freq
    signal_wave = wave_object.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)

    if channel != 1:
        raise Exception("Some sample have more than one channel")
    times = np.linspace(0, t_audio, num=n_samples)
    plt.figure(figsize=(15, 5))
    plt.plot(times, signal_array)
    plt.title("Audio Wave")
    plt.ylabel("Signal Value")
    plt.xlabel("Time (s)")
    plt.xlim(0, t_audio)
    plt.grid()
    plt.savefig(destination_file)
    plt.close()
    # plt.show()
