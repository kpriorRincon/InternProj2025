import numpy as np

# Define start and end sequences
# 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 0 0 0 1 0 1 0 1 1 1 0 1 1 0 0 0 1
start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                1, 0, 1, 0, 0, 1, 0, 0,
                0, 0, 1, 0, 1, 0, 1, 1,
                1, 0, 1, 1, 0, 0, 0, 1]
print("Start sequence: ", start_sequence)

# 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0 1 0 0 1 0
end_sequence = [0, 0, 1, 0, 0, 1, 1, 0,
                1, 0, 0, 0, 0, 0, 1, 0, 
                0, 0, 1, 1, 1, 1, 0, 1, 
                0, 0, 0, 1, 0, 0, 1, 0]
print("End sequence: ", end_sequence)

######################### Functions #########################
def bits_to_symbols(bits):
    return [int(''.join(map(str, bits[i:i+2])), 2) for i in range(0, len(bits), 2)]

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            continue
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)
######################## Main Program #######################
# Clear files
open('bits_to_send.bin', 'w').close()
open('bits_read_in.bin', 'w').close()

# Get input
msg = input("Enter your message: ")
print("Message length: ", len(msg))

# Convert to bits
bits = [int(b) for c in msg for b in format(ord(c), '08b')]
print("Message in binary: ", bits)

# Add start and end markers
bits = start_sequence + bits + end_sequence
print("Total number of bits to send: ", len(bits))

# error check the start and end markers in text
text_output = bits_to_text(bits)
print("Text being sent: ", text_output)

# save to file
np.array(bits, dtype=np.uint8).tofile("bits_to_send.bin")
read_input = np.fromfile("bits_to_send.bin", dtype=np.uint8)
print("What was written to file:\n", read_input)
