import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rtlsdr import RtlSdr

# Configure RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6       # Hz
sdr.center_freq = 30e6       # Hz (FM band)
sdr.gain = 'auto'

# Spectrogram parameters
fft_size = 1024
hop_size = fft_size // 2
spec_history = 100  # number of lines in spectrogram

# Prepare plot
fig, ax = plt.subplots()
spec_data = np.zeros((spec_history, fft_size // 2))
img = ax.imshow(spec_data, aspect='auto', origin='lower',
                extent=[0, sdr.sample_rate / 2 / 1e6, 0, spec_history],
                cmap='viridis', vmin=-100, vmax=0)

ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Time (frames)")

# Update function
def update(frame):
    samples = sdr.read_samples(fft_size)
    windowed = samples * np.hanning(fft_size)
    spectrum = np.fft.fft(windowed)
    power = 20 * np.log10(np.abs(spectrum[:fft_size // 2]) + 1e-12)

    global spec_data
    spec_data = np.roll(spec_data, -1, axis=0)
    spec_data[-1, :] = power
    img.set_data(spec_data)
    return [img]

ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
plt.tight_layout()
plt.show()

sdr.close()
