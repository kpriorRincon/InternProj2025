import numpy as np
import matplotlib.pyplot as plt
import Sig_Gen as SigGen

# Generate a sine wave
fs = 1e6  # Sampling frequency
f = 10e3  # Sine frequency
duration = 1e-3  # seconds
t = np.arange(0, duration, 1/fs)
sine_wave = np.sin(2 * np.pi * f * t)

# Plot the sine wave
plt.plot(t, sine_wave)
plt.title("Sine Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()


# Apply time delay filter
N = 21
delay = 0.4
n = np.arange(-N//2, N//2)
h = np.sinc(n - delay)
h *= np.hamming(N)
h /= np.sum(h)  # unit gain

# Filter the sine wave
sine_delayed = np.convolve(sine_wave, h, mode='same')

# Plot delayed sine wave on the same plot
plt.plot(t, sine_delayed, label='Delayed Sine')
plt.plot(t, sine_wave, label='Original Sine', alpha=0.7)
plt.title("Sine Wave with Time Delay Filter")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Plot frequency domain
freqs = np.fft.fftfreq(len(sine_wave), 1/fs)
S_orig = np.fft.fft(sine_wave)
S_delayed = np.fft.fft(sine_delayed)

plt.plot(freqs[:len(freqs)//2], np.abs(S_orig[:len(freqs)//2]), label='Original')
plt.plot(freqs[:len(freqs)//2], np.abs(S_delayed[:len(freqs)//2]), label='Delayed')
plt.title("Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.show()