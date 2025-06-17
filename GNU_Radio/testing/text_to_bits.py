import numpy as np

# Clear files
open('bits_to_send.txt', 'w').close()
open('bits_read_in.txt', 'w').close()

# Get input
message = input("Enter your message: ")
print("Transmitted message length: ", len(message))

# Convert to bits
bits = [int(bit) for c in message for bit in format(ord(c), '08b')]
print("Transmitted bits length: ", len(bits))
print("First 32 bits sent:", bits[:32])

# Save to file
np.array(bits, dtype=np.uint8).tofile("bits_to_send.bin")
