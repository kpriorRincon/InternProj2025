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

f_cutoff = 930e6
f_sample = 10 * f_cutoff  # 9.3 GHz
order = 5

# Time vector
duration = 5 / 1.83e9  # 5 cycles of f_lo
t = np.arange(0, duration, 1 / f_sample)

# Mixed signal
f_lo = 1.83e9
f_in = 910e6
sig = np.cos(2 * np.pi * f_in * t) * np.cos(2 * np.pi * f_lo * t)

# Design Butterworth LPF
b, a = signal.butter(order, f_cutoff, btype='low', fs=f_sample)

# Apply filter
filtered_sig = signal.filtfilt(b, a, sig)

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

