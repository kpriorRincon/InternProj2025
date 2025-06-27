from rtlsdr import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import Detector as d
import receive_processing as rp
import time
import transmit_processing as tp

# configure RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6 # Hz
sdr.center_freq = 11e6 # Hz
sdr.freq_correction = 60 # PPM
sdr.gain = 'auto'

# sleep
time.sleep(1)

# settings to run detector
N = 1024
detected = False
start = 0
end = N - 1
sps = 4
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

# plot settings - MODIFIED
def update_plot(data):
    f, t, Sxx = signal.spectrogram(data, fs=sdr.sample_rate)
    # Convert frequency bins to actual frequencies centered around SDR center frequency
    f_actual = f - sdr.sample_rate/2 + sdr.center_freq
    return f_actual, t, Sxx

# Initialize plot with proper frequency scaling
f, t, Sxx_init = update_plot(np.random.randn(N) * 0.01)
ax = plt.subplot(1,1,1)
im = ax.imshow(10*np.log10(Sxx_init), aspect='auto', origin='lower', 
               extent=[t.min(), t.max(), f.min()/1e6, f.max()/1e6])  # Convert to MHz
plt.colorbar(im, label='Power (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (MHz)')
plt.title(f'Spectrogram - Center: {sdr.center_freq/1e6:.1f} MHz')
plt.ion()

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
    
    # plot samples - MODIFIED
    f, t, Sxx = update_plot(samples)
    im.set_data(10*np.log10(Sxx))
    im.set_extent([t.min(), t.max(), f.min()/1e6, f.max()/1e6])  # Update extent
    plt.draw()
    plt.pause(0.01)
    
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