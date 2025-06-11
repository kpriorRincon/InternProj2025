#################
#
# Author: Trevor
#
#################

# imports
from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt

# configure the RTL SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6 # Hz
sdr.center_freq = 920e6   # Hz
sdr.freq_correction = 60  # PPM
print(sdr.valid_gains_db)
sdr.gain = 49.6
print(sdr.gain)

# run the receiver code
