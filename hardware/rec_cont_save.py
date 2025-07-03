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
open('test_data.bin', 'wb').close()  # clear the file before writing
while count < 10:
    count += 1  # increment cycle count
    # read samples from RTL-SDR
    samples = None
    samples = sdr.read_samples(N)

    # save samples to an external file (optional) 
    with open('test_data.bin', 'ab') as f:
        # Convert samples to complex64 and write to file
        f.write(np.array(samples, dtype=np.complex64).tobytes())
    
    strt_t = time.time()
    # run detection
    
    total_t = time.time() - strt_t
    #detected, coarse_fixed = detect_obj.detector(samples, match_start=match_start, match_end=match_end)

# error check
print("Done saving samples")
raw_data = np.fromfile("test_data.bin", dtype=np.complex64)
print("Length of data: ", len(raw_data))
print("Total time: ", total_t)

# close sdr
sdr.close()