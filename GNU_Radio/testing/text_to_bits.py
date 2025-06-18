import numpy as np
import modulator as md
from scipy.signal import fftconvolve

def bits_to_symbols(bits):
    return [int(''.join(map(str, bits[i:i+2])), 2) for i in range(0, len(bits), 2)]

# Clear files
open('bits_to_send.txt', 'w').close()
open('bits_read_in.txt', 'w').close()

# Get input
msg = input("Enter your message: ")
print("Message length: ", len(msg))

# Convert to bits
bits = [int(b) for c in msg for b in format(ord(c), '08b')]
print("Total bits to send: ", len(bits))

# create qpsk symbols
symbols = md.generate_qpsk(bits)

# upsample symbols
sps = 2
upsampled_symbols = np.concatenate([np.append(x, np.zeros(sps-1))for x in symbols])

# pulse shaping
beta = 0.35
N = 64
Ts = 1.0
fs = 32000
h = md.rrc_filter(beta, N, Ts, fs)

# Convolve symbols with the filter
shaped_symbols = fftconvolve(upsampled_symbols, h, mode='same')

# normalize symbols
symbols /= np.max(np.abs(symbols))

# error check
print("First 32 bits sent:", bits[:32])

# Save to file
#np.array(symbols, dtype=np.uint8).tofile("bits_to_send.bin")
np.array(symbols, dtype=np.complex64).tofile("bits_to_send.bin")
read_input = np.fromfile("bits_to_send.bin", dtype=np.complex64)
print("What was written to file:\n", read_input)
