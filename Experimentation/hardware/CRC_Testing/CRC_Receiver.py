from rtlsdr import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.signal as signal
import Detector as d
import time
import hardware.CRC_Testing.CRC_Transmit_Processing as tp
from channel_correction import *
from config import *
from crc import Calculator, Crc8

# configure RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE # Hz
sdr.center_freq = RX_REC_FREQ # Hz
sdr.freq_correction = PPM # PPM
sdr.gain = 'auto'

# initialize the CRC
calculator = Calculator(Crc8.CCITT)

# sleep
time.sleep(1)

# settings to run detector
detected = False
sps = SPS
N = sps * 1024
start = 0
end = N - 1
beta = BETA
num_taps = NUMTAPS
symbol_rate = sdr.sample_rate / sps

# create transmit object for the start and end markers
transmit_obj = tp.transmit_processing(sps, sdr.sample_rate)
match_start, match_end = transmit_obj.modulated_markers(beta, num_taps) 

# detector object
detect_obj = d.Detector(sdr.sample_rate)

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
data = coarse_fixed
print(f"Signal found after {count} cycles")

# begin signal processing
print("Processing data...")                                                 # where are we in the code
strt_t = time.time()                                                        # how long does this take
bits_string, decoded_message = channel_handler(data)                        # process the signal and decode the message
total_t = time.time() - strt_t
print(f"Time to run rest of RX chain to till demod: {total_t} s")

# CRC Check
byte_data = int(bits_string, 2).to_bytes((len(bits_string) + 7) // 8, 'big')# convert the bit string to bytes
check = calculator.checksum(byte_data)

print("Remainder: ", check)
if check == 0:
    data = byte_data[:-2].decode('ascii')
    print("Data is valid...")
    print(f"Bits: {bits_string}")
    print(f"Message: {data}")
else:
    print("Data is invalid...\nAborting...")

# close sdr
sdr.close()
