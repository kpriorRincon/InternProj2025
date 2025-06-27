from rtlsdr import *
import matplotlib.pyplot as plt
import numpy as np
import Detector as d
import receive_processing as rp

# configure RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = 2.048e6  # Hz
sdr.center_freq = 88.1e6     # Hz
sdr.freq_correction = 60   # PPM
sdr.gain = 'auto'

# settings to run detector
N = 2056
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

# detector object
detect_obj = d.Detector(start_marker, end_marker, beta, N, 1 / symbol_rate, sdr.sample_rate)

# create matched filters
match_start, match_end = detect_obj.matchedFilter(sps)

# run detection
while detected == False:
    samples = sdr.read_samples(N)
    detected, start, end = detect_obj.detector(samples, match_start=match_start, match_end=match_end)
data = samples[start:end]

print("Signal found...")

# close sdr
sdr.close()

# begin signal processing
print("Processing data...")

# time correction

# Phase correction

# frequency correction

# create receive processing object

recieve_obj = rp.receive_processing(sps, sdr.sample_rate)

# process data
recieve_obj.work(data, beta, num_taps)