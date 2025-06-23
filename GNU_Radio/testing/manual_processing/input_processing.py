import numpy as np
import modulator as md
from scipy.signal import fftconvolve
from commpy.filters import rrcosfilter
from matplotlib import pyplot as plt
import qpsk_signal_processing as qpsk

####################### User input to symbols ###########################
# Clear files
open('bits_to_send.bin', 'w').close()
open('bits_read_in.bin', 'w').close()

# User input
msg = input("Enter your message: ")

# Convert to bits
bits = [int(b) for c in msg for b in format(ord(c), '08b')]
print("Bits to send\n", bits)

# Create QPSK symbols
symbols = md.generate_qpsk(bits)
#print(f"Number of QPSK symbols: {len(symbols)}")
#plt.scatter(np.real(symbols), np.imag(symbols), marker='o')

####################### Signal Processing ##########################
# Filter parameters
sps = 2             # samples per symbol
beta = 0.35         # roll off 
taps = 31           # span = 11 symbols
samp_rate = 32000   # sampling rate

# Create RRC filter
h, _ = rrcosfilter(taps, beta, sps / samp_rate, samp_rate)
print(f"RRC Filter length: {len(h)}")

# Zero-insertion upsampling
upsampled_symbols = np.zeros(len(symbols) * sps, dtype=np.complex64)
upsampled_symbols[::sps] = symbols
# upsampled_symbols =  np.repeat(symbols, sps)

#print("What the symbols look like \n", upsampled_symbols)
#plt.scatter(np.real(upsampled_symbols), np.imag(upsampled_symbols), marker='o')

# Pulse shaping
shaped_symbols = fftconvolve(upsampled_symbols, h, mode='full')

# Normalize symbols
max_val = np.max(np.abs(shaped_symbols))
if max_val > 0:
    shaped_symbols /= max_val

# print("What the symbols look like \n", shaped_symbols)
# plt.scatter(np.real(shaped_symbols), np.imag(shaped_symbols), marker='o')
# plt.grid()
# plt.title("Transmitted Symbols")
# plt.show()

######################## Save to Transmit File ######################
np.array(shaped_symbols, dtype=np.complex64).tofile("bits_to_send.bin")
read_input = np.fromfile("bits_to_send.bin", dtype=np.complex64)
print("Shape of what was written to file:", read_input.shape)
