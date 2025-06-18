import numpy as np

def bits_to_symbols(bits):
    return [int(''.join(map(str, bits[i:i+2])), 2) for i in range(0, len(bits), 2)]

# Clear files
open('bits_to_send.txt', 'w').close()
open('bits_read_in.txt', 'w').close()

# Get input
msg = input("Enter your message: ")

# Convert to bits
bits = [int(b) for c in msg for b in format(ord(c), '08b')]

# error check
symbols = bits_to_symbols(bits)
print("First 16 symbols sent:", symbols[:16])
print("First 32 bits sent:", bits[:32])

# Save to file
np.array(bits, dtype=np.uint8).tofile("bits_to_send.bin")
