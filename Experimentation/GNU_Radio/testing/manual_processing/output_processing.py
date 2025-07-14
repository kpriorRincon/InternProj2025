import numpy as np
import demodulator as dm
from scipy.signal import fftconvolve
from commpy.filters import rrcosfilter
from matplotlib import pyplot as plt
import correlator as cr

#################################### Helper Functions #############################
def bits_to_text(bits):
    """ Convert bitstream to ASCII text """
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            continue
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

#################################### Load Received Data ###########################
raw_data = np.fromfile("bits_read_in.bin", dtype=np.complex64)
print("Raw data shape:", raw_data.shape)

#################################### Signal Processing ############################
# Filter parameters
sps = 2             # samples per symbol
beta = 0.35         # roll off 
taps = 31           # must match Tx
samp_rate = 32000   # sampling rate
decimation_factor = sps

# RRC matched filter
_, h = rrcosfilter(taps, beta, sps / samp_rate, samp_rate)
print(f"RRC Filter length: {len(h)}")

# Matched filtering
filtered_symbols = fftconvolve(raw_data, np.conj(h), mode='full')
print("Filtered signal length: ", len(filtered_symbols))

# Correlation
start, end = cr.correlator(filtered_symbols, sps)
print(f"Start {start}, End {end}")
correlated_signal = filtered_symbols[start:end]
#print("Correlated signal: \n", correlated_signal)

# Remove group delay and downsample
group_delay = int((taps-1)/2)
symbols = correlated_signal[::sps]
print("Number of symbols after decimation: ", len(symbols))

# Normalize
#max_val = np.max(np.abs(symbols))
#if max_val > 0:
#    symbols /= max_val

##################################### Demodulation ##########################
if len(symbols) > 0:
    bits = dm.phase_rotation_handler(symbols)

    print("First 32 bits received:", bits)
    print("Total bits received: ", len(bits))

    # Convert to text
    text = bits_to_text(bits)
    print(f"Decoded message: '{text}'")
    print("Received message length: ", len(text))
else:
    print("No symbols recovered!")

# print("Received symbols:\n", symbols)
# plt.scatter(np.real(symbols), np.imag(symbols), marker='o')
# plt.grid()
# plt.title("Received Symbols")
# plt.show()
