import numpy as np
import demodulator as dm
from scipy.signal import fftconvolve

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            continue
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

# Load file
#raw_data = np.fromfile("bits_read_in.bin", dtype=np.uint8)
raw_data = np.fromfile("bits_read_in.bin", dtype=np.complex64)
print("Raw data loaded from file:\n", raw_data)

# pulse shaping
# beta = 0.35
# N = 64
# Ts = 1.0
# fs = 32000
# h = dm.rrc_filter(beta, N, Ts, fs)

# # Convolve symbols with the filter
# symbols = fftconvolve(raw_data, h, mode='same')

# # normalize symbols
# symbols /= np.max(np.abs(symbols))

# downsample symbols
# sps = 2
# symbols = symbols[::sps]

#bits = dm.read_qpsk(symbols)
bits = dm.read_qpsk(raw_data)
print("First 32 bits received:", np.flip(bits[:32]))
print("Total bits received: ", len(bits))

# print the received message
text = bits_to_text(bits)
print(f"Decoded message: '{text}'")
print("Received message length: ", len(text))
