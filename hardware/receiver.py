from rtlsdr import *
import matplotlib.pyplot as plt
import numpy as np
import Detector

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

# detector object
detect_obj = Detector.Detector()

# create matched filters
match_start, match_end = detect_obj.matchedFilter(sps)

# run detection
while detected == False:
    samples = sdr.read_samples(N)
    detected, start, end = detect_obj.detector(samples, match_start=match_start, match_end=match_end)

print("Signal found...")

# close sdr
sdr.close()

# begin signal processing
print("Processing data...")

# time correction

# Phase correction

# frequency correction

# anti aliasing filter

# decimation

# demodulation

# read out message