from rtlsdr import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.signal as signal
import Detector as d
import receive_processing as rp
import time
import transmit_processing as tp
from channel_correction import *

# configure RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = 2.88e6 # Hz
sdr.center_freq = 920e6 # Hz
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

# create transmit object for the start and end markers
transmit_obj = tp.transmit_processing(sps, sdr.sample_rate)
match_start, match_end = transmit_obj.modulated_markers(beta, num_taps) 

# detector object
detect_obj = d.Detector(sdr.sample_rate)

# Test SDR connection before main loop
try:
    test_samples = sdr.read_samples(1024)
    print(f"SDR test successful: read {len(test_samples)} samples")
except Exception as e:
    print(f"SDR test failed: {e}")
    sdr.close()
    exit()

total_t = 0
# run detection
count = 0   # count cycles until detected
open('test_data.bin', 'a')
while detected == False:
    count += 1  # increment cycle count
    # read samples from RTL-SDR
    samples = None
    samples = sdr.read_samples(N)

    # save samples to an external file (optional) 
    np.array(samples, dtype=np.complex64).tofile("test_data.bin")
    strt_t = time.time()
    # run detection
    
    total_t = time.time() - strt_t
    detected, coarse_fixed = detect_obj.detector(samples, match_start=match_start, match_end=match_end)

print(f"Time to run detection on buffer: {total_t} s")
# take signal from the samples
#data = samples[start:end]
data = coarse_fixed
# open('selected_signal.bin', 'w').close()
# np.array(data, dtype=np.complex64).tofile("selected_signal.bin")
print(f"Signal found after {count} cycles")

# begin signal processing
print("Processing data...")

strt_t = time.time()
bits_string, decoded_message = channel_handler(data)
# create receive processing object
# recieve_obj = rp.receive_processing(sps, sdr.sample_rate)
total_t = time.time() - strt_t
print(f"Time to run rest of RX chain to till demod: {total_t} s")
# process data
# bits_string, message = recieve_obj.work(data, beta, num_taps)
print(f"Bits: {bits_string}")
print(f"Message: {decoded_message}")

# close sdr
sdr.close()
