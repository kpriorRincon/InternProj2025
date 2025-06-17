import numpy as np

def bits_already_binary(bit_values):
    """For when the file already contains binary bits (0s and 1s)"""
    return [int(bit) for bit in bit_values]

def get_string(bits):
    """Convert binary bits to ASCII string"""
    ascii_chars = []
    
    # Process 8 bits at a time
    for i in range(0, len(bits), 8):
        # Get 8 bits
        byte_bits = bits[i:i+8]
        
        # Ensure we have exactly 8 bits
        if len(byte_bits) < 8:
            break  # Skip incomplete bytes
            
        # Convert to binary string
        byte_str = ''.join(map(str, byte_bits))
        byte_value = int(byte_str, 2)

        # Make ASCII characters
        ascii_chars.append(chr(byte_value))

    # Return string of ASCII
    return ''.join(ascii_chars)

# Read raw symbol values (0–3)
raw_data = np.fromfile("bits_read_in.bin", dtype=np.uint8)
print("Received symbols length: ", len(raw_data))

# Convert 2-bit symbols to bitstream
bitstream = []
for symbol in raw_data:
    bits = format(symbol, '02b')  # 2-bit binary
    bitstream.extend([int(b) for b in bits])

print("First 32 bits received:", bitstream[:32])

# Decode to ASCII
output = get_string(bitstream)
print(f"Decoded message: '{output}'")
print("Received message length: ", len(output))
