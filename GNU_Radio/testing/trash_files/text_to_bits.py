import numpy as np

# Define start and end sequences
# 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 0 0 0 1 0 1 0 1 1 1 0 1 1 0 0 0 1
start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                1, 0, 1, 0, 0, 1, 0, 0,
                0, 0, 1, 0, 1, 0, 1, 1,
                1, 0, 1, 1, 0, 0, 0, 1]

# 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0 1 0 0 1 0
end_sequence = [0, 0, 1, 0, 0, 1, 1, 0,
                1, 0, 0, 0, 0, 0, 1, 0, 
                0, 0, 1, 1, 1, 1, 0, 1, 
                0, 0, 0, 1, 0, 0, 1, 0]

def bits_to_symbols(bits):
    return [int(''.join(map(str, bits[i:i+2])), 2) for i in range(0, len(bits), 2)]

# Clear files
open('bits_to_send.bin', 'w').close()
open('bits_read_in.bin', 'w').close()

# Get input
msg = input("Enter your message: ")
print("Message length: ", len(msg))

# Convert to bits
bits = [int(b) for c in msg for b in format(ord(c), '08b')]
print("Total bits to send: ", len(bits))

# error check
print("Message in binary:", bits)

# Save to file
bits = start_sequence + bits + end_sequence
np.array(bits, dtype=np.uint8).tofile("bits_to_send.bin")
read_input = np.fromfile("bits_to_send.bin", dtype=np.uint8)
print("What was written to file:\n", read_input)
