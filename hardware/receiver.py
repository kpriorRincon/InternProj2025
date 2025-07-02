from rtlsdr import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.signal as signal
import Detector_copy as d
import receive_processing as rp
import time
import transmit_processing as tp

# configure RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = 2.88e6 # Hz
sdr.center_freq = 905e6 # Hz
sdr.freq_correction = 60 # PPM
sdr.gain = 'auto'

# sleep
time.sleep(1)

# settings to run detector
detected = False
sps = 20
N = 20 * 1024
start = 0
end = N - 1
beta = 0.35
num_taps = 101
symbol_rate = sdr.sample_rate / sps

transmit_obj = tp.transmit_processing(sps, sdr.sample_rate)
match_start, match_end = transmit_obj.modulated_markers(beta, num_taps) 

# detector object
detect_obj = d.Detector(N, 1 / symbol_rate, beta, sdr.sample_rate, sps=sps)

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
    samples = None
    samples = sdr.read_samples(N)
    #detect_obj.step_detect(samples)      
    # plot samples
    plt.plot(np.real(samples))
    plt.plot(np.imag(samples))
    plt.show()
    
    #np.array(samples, dtype=np.complex64).tofile("test_data.bin")

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
