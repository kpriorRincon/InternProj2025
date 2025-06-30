from rtlsdr import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.signal as signal
import Detector as d
import receive_processing as rp
import time
import transmit_processing as tp

# configure RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6 # Hz
sdr.center_freq = 30e6 # Hz
sdr.freq_correction = 60 # PPM
sdr.gain = 'auto'

# sleep
time.sleep(1)

# settings to run detector
detected = False
sps = 10
N = sps * 1024
start = 0
end = N - 1
beta = 0.35
num_taps = 40
symbol_rate = sdr.sample_rate / sps

# markers
start_marker = [1, 1, 1, 1, 1, 0, 0, 1,
                1, 0, 1, 0, 0, 1, 0, 0,
                0, 0, 1, 0, 1, 0, 1, 1,
                1, 0, 1, 1, 0, 0, 0, 1]

end_marker = [0, 0, 1, 0, 0, 1, 1, 0,
              1, 0, 0, 0, 0, 0, 1, 0,
              0, 0, 1, 1, 1, 1, 0, 1,
              0, 0, 0, 1, 0, 0, 1, 0]

transmit_obj = tp.transmit_processing(sps, sdr.sample_rate)
start_symbols = transmit_obj.qpsk_mapping(start_marker)
end_symbols = transmit_obj.qpsk_mapping(end_marker)

# detector object
detect_obj = d.Detector(start_symbols, end_symbols, N, 1 / symbol_rate, beta, sdr.sample_rate, sps=2)

# create matched filters
match_start, match_end = detect_obj.matchedFilter(sps)

# Spectrogram parameters
fft_size = N
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
ax.set_title("Baseband Spectrogram of RTL-SDR ")

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

# Test SDR connection before main loop
try:
    test_samples = sdr.read_samples(1024)
    print(f"SDR test successful: read {len(test_samples)} samples")
except Exception as e:
    print(f"SDR test failed: {e}")
    sdr.close()
    exit()

# run detection
count = 0
while detected == False:
    count += 1
    
    # read samples from RTL-SDR
    samples = sdr.read_samples(N)
    
    # plot samples
    
    # run detection
    detected, start, end = detect_obj.detector(samples, match_start=match_start, match_end=match_end)

data = samples[start:end]
print(f"Signal found after {count} cycles")

# plot handling
plt.ioff()
plt.show()

# begin signal processing
print("Processing data...")

# time correction
# Phase correction  
# frequency correction

# create receive processing object
recieve_obj = rp.receive_processing(sps, sdr.sample_rate)

# process data
bits_string, message = recieve_obj.work(data, beta, num_taps)
print(f"Bits: {bits_string}")
print(f"Message: {message}")

# close sdr
sdr.close()
