import numpy as np

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            continue
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

# Load file
raw_data = np.fromfile("bits_read_in.bin", dtype=np.uint8)
print("Raw data loaded from file:\n", raw_data)

# Conver raw data to bits
bits = raw_data.tolist()
print("First 32 bits received:", bits[:32])
print("Total bits received: ", len(bits))

# print the received message
text = bits_to_text(bits)
print(f"Decoded message: '{text}'")
print("Received message length: ", len(text))

bit_string = ''.join(map(str, bits))
# Write bits and text to .txt file
with open('bits_read_in.txt', 'w') as txt_file:
    txt_file.write(bit_string)
    txt_file.write('\n')
    txt_file.write(text)
