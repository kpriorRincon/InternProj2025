from rtlsdr import *
import matplotlib.pyplot as plt
import numpy as np
import SignalProcessing as sp

sdr = RtlSdr()

sdr.sample_rate = 2.048e6  # Hz
sdr.center_freq = 88.1e6     # Hz
sdr.freq_correction = 60   # PPM
sdr.gain = 'auto'

N = 2056
detected = False
start = 0
end = N - 1

while detected == False:
    samples = sdr.read_samples(N)
    detected, start, end = sp.detector(samples)