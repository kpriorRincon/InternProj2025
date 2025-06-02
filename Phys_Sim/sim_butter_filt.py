import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

#####################################################
#
# Author: Trevor Wiseman
#
#####################################################

def mix_signal(f_lo, f_in, f_cutoff):
    # Sampling frequency
    f_sample = 10 * f_cutoff    

    # Time vector
    N = 2000
    t = np.arange(N) / f_sample    # time vector
    
    # Mixed signal
    sig = np.cos(2 * np.pi * f_in * t) * np.cos(2 * np.pi * f_lo * t)   # Mixed Signal

    return t, sig, f_sample

def filter_the_signal(f_cutoff = None, order=5, sig = None, f_sample = None):
    # Design Butterworth LPF
    b, a = signal.butter(order, f_cutoff, btype='low', fs=f_sample) # butterworth filter coefficients

    # Apply filter
    filtered_sig = signal.filtfilt(b, a, sig)   # filtered signal

    return filtered_sig

#### Test the Function #####
# f_cutoff = 930e6            # Cutoff frequency
# order = 5                   # Filter order
# f_lo = 1.83e9   # Local Oscillator
# f_in = 910e6    # Input frequency

# # mix the signal
# t, sig, f_sample = mix_signal(f_lo, f_in)  # Generate mixed signal
# # filter the signal
# t, sig, filtered_sig, f_sample = filter_the_signal(f_cutoff, 5, sig, f_sample)

# fft_sig = np.abs(fft(sig))  # Optional: Compute FFT of the original signal for analysis
# fft_filtered = np.abs(fft(filtered_sig))  # Optional: Compute FFT of the filtered signal for analysis

# freq_sig = fftfreq(len(t), 1/f_sample)  # Frequency bins for the original signal
# freq_filtered = fftfreq(len(t), 1/f_sample)  # Frequency bins for the filtered signal
# print("length ", len(t))

# # Plot time domain signals
# plt.figure()
# plt.plot(t, sig, label='Original Signal')
# plt.plot(t, filtered_sig, label='Filtered Signal', linestyle='--')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Low-pass Filtered Signal')
# plt.legend()
# plt.grid()
# plt.show()

# # plot fft of the signals
# plt.figure()
# plt.plot(freq_sig, np.abs(fft_sig), label='FFT of Original Signal')
# plt.plot(freq_filtered, np.abs(fft_filtered), label='FFT of Filtered Signal', linestyle='--')
# plt.xlabel('Frequency (GHz)')
# plt.ylabel('Magnitude')
# plt.title('FFT of Signals')
# plt.xlim(0, 3e9)  # Limit x-axis to 2 GHz for better visibility
# plt.legend()
# plt.grid()
# plt.show()