import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
from scipy.io import wavfile

def plot_time_signal_seconds(signal,sampling_freq):
    time_in_sec = signal.shape[0] / sampling_freq # get sound duration in sec
    time  = np.arange(signal.shape[0]) / signal.shape[0] * time_in_sec 

    # plt.subplot(2,1,1)
    plt.plot(time , signal[:], 'r')
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.show()
    # plt.plot(time[40000:45000], sound[40000:45000])
    # plt.xlabel("time, s")
    # plt.ylabel("")
    # plt.show()

def get_fft(signal, sampling_freq):
    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1./sampling_freq)
    return fft_spectrum, freq


def plot_fft_spectrum(signal, sampling_freq):
    fft_spectrum, freq = get_fft(signal, sampling_freq)
    # As we need the magnitude of our FFT transformation not the phase, then get the abs of fft_sepctrum
    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.plot(freq, fft_spectrum_abs)
    plt.xlabel("Frequency | Hz")
    plt.ylabel("Amplitude")
    plt.show()


def play_audio(signal, sampling_freq):
    aud = signal[0]
    sd.play(signal, sampling_freq)
    sd.wait()

# main program goes here
def main():
    # sampling_freq represents how many samples are taken in 1 sec
    sampling_freq, signal = wavfile.read('./mina.wav')
    # make samples magnitudes varies from -1,1
    signal = signal / 2.0**15 
    fft_sepctrum, freq = get_fft(signal, sampling_freq)
    signal1 = np.fft.irfft(fft_sepctrum)
    print("signal1" , signal1)
    print("signal" , signal)
    plot_time_signal_seconds(signal, sampling_freq) # Time Domain Plot
    plot_fft_spectrum(signal, sampling_freq)      # Frequency Domain Plot
    # play_audio(signal, sampling_freq)   # Original signal
    # play_audio(signal1, sampling_freq)  # Reversed FFT signal
    

if __name__  == '__main__':
    main()

