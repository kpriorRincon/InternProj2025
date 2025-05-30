import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt # For visualization

# 1. Parameters
sample_rate = 1000 # Hz
lowcut = 100      # Hz
highcut = 300     # Hz
order = 5          # Filter order

# 2. Generate a sample signal (replace with your actual signal)
duration = 1     # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
sig = 0.5 * np.sin(2*np.pi*60*t) + np.sin(2*np.pi*250*t) + 0.2 * np.sin(2*np.pi*500*t) # Mixed signal

# 3. Design the bandpass filter
sos = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=sample_rate, output='sos')

# 4. Apply the filter
filtered_sig = signal.sosfiltfilt(sos, sig)

# 5. Visualize
plt.figure()
plt.plot(t, sig, label='Original Signal')
plt.plot(t, filtered_sig, label='Filtered Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
