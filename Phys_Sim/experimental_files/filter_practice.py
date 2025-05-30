# import numpy as np
# import scipy.signal as signal
# import matplotlib.pyplot as plt # For visualization

# # 1. Parameters
# lowcut = 930e6      # Hz
# highcut = 1000e6     # Hz
# sample_rate = 10*highcut # Hz
# order = 5          # Filter order

# # 2. Generate a sample signal (replace with your actual signal)
# f_lo = 1.83e9
# f_in = 910e6
# duration = 5 / f_lo    # seconds
# t = np.linspace(0, duration, 150, endpoint=False)
# sig = np.cos(2*np.pi*f_in*t)*np.cos(2*np.pi*f_lo*t) # Mixed signal

# # 3. Design the bandpass filter
# sos = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=sample_rate, output='sos')

# # 4. Apply the filter
# filtered_sig = signal.sosfiltfilt(sos, sig)

# # 5. Visualize
# plt.figure()
# plt.plot(t, sig, label='Original Signal')
# plt.plot(t, filtered_sig, label='Filtered Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()


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
f_cutoff = 930e6            # Cutoff frequency
order = 5                   # Filter order
f_lo = 1.83e9   # Local Oscillator
f_in = 910e6    # Input frequency

t, sig, filtered_sig = filter_the_signal(f_cutoff, f_lo, f_in, order)

# Plot
plt.figure()
plt.plot(t, sig, label='Original Signal')
plt.plot(t, filtered_sig, label='Filtered Signal', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Low-pass Filtered Signal')
plt.legend()
plt.grid()
plt.show()

