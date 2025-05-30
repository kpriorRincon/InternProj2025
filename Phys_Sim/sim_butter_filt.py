import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def filter_the_signal(f_cutoff, f_lo, f_in, order=5):
    # Sampling frequency
    f_sample = 10 * f_cutoff    

    # Time vector
    duration = 20 / f_lo                      # number of cycles of f_lo
    t = np.arange(0, duration, 1 / f_sample)    # time vector

    # Mixed signal
    sig = np.cos(2 * np.pi * f_in * t) * np.cos(2 * np.pi * f_lo * t)   # Mixed Signal

    # Design Butterworth LPF
    b, a = signal.butter(order, f_cutoff, btype='low', fs=f_sample) # butterworth filter coefficients

    # Apply filter
    filtered_sig = signal.filtfilt(b, a, sig)   # filtered signal

    return t, sig, filtered_sig

##### Test the Function #####
# f_cutoff = 930e6            # Cutoff frequency
# order = 5                   # Filter order
# f_lo = 1.83e9   # Local Oscillator
# f_in = 910e6    # Input frequency

# t, sig, filtered_sig = filter_the_signal(f_cutoff, f_lo, f_in, order)

# # Plot
# plt.figure()
# plt.plot(t, sig, label='Original Signal')
# plt.plot(t, filtered_sig, label='Filtered Signal', linestyle='--')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Low-pass Filtered Signal')
# plt.legend()
# plt.grid()
# plt.show()