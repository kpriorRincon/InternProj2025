import numpy as np
import matplotlib.pyplot as plt

def bit_reader(symbols):
    bits = np.zeros((len(symbols), 2), dtype=int)
    for i in range(len(symbols)):
        angle = np.angle(symbols[i], deg=True) % 360

        # grey coding codex
        if 0 <= angle < 90:
            bits[i] = [1, 1]
        elif 90 <= angle < 180:
            bits[i] = [0, 1]
        elif 180 <= angle < 270:
            bits[i] = [0, 0]
        else:  # 270 <= angle < 360
            bits[i] = [1, 0]
    return bits

def get_string(bits):
    ascii_chars = []
    
    # process 8 bits at a time
    for i in range(0, len(bits), 8):
        # get 8 bits
        byte_bits = bits[i:i+8]
        
        # ensure we have exactly 8 bits
        if len(byte_bits) < 8:
            break  # skip incomplete bytes
            
        # convert to binary string
        byte_str = ''.join(map(str, byte_bits))
        byte_value = int(byte_str, 2)

        # make ascii characters
        ascii_chars.append(chr(byte_value))

    # return string of ascii
    return ''.join(ascii_chars)

# Read complex symbols from file
f = np.fromfile("bits_read_in.txt", dtype=np.float64)
print(f"Symbols: {f}")

# Optional: plot constellation
# plt.scatter(np.real(f), np.imag(f))
# plt.show()

# Decode symbols to bits
bits = bit_reader(f)
print(f"Bits: {bits}")

# Flatten bits to 1D array
flat_bits = bits.flatten()
print(f"Flat bits: {flat_bits}")

# Convert to string
output = get_string(flat_bits)
print(f"Decoded message: '{output}'")