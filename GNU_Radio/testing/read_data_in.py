import numpy as np

def decimal_to_bits(decimal_values):
    """Convert array of decimal values to binary bits"""
    all_bits = []
    for decimal in decimal_values:
        # Convert decimal to 8-bit binary string
        bit_str = format(int(decimal), '08b')
        # Convert string to list of integers
        bits = [int(b) for b in bit_str]
        all_bits.extend(bits)
    return all_bits

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

# For GNU Radio binary files
try:
    # Read as unsigned 8-bit integers
    raw_data = np.fromfile("bits_read_in.txt", dtype=np.uint8)
    print(f"File size: {len(raw_data)} bytes")
    print(f"First 20 values: {raw_data[:20]}")
    
    # Check if data is already binary (only 0s and 1s)
    unique_values = np.unique(raw_data)
    print(f"Unique values in file: {unique_values}")
    
    if len(unique_values) <= 2 and all(val in [0, 1] for val in unique_values):
        print("Data appears to be binary bits (0s and 1s)")
        # Use the data directly as bits
        bits = bits_already_binary(raw_data)
        print(f"Total bits: {len(bits)}")
        print(f"First 40 bits: {bits[:40]}")
        
        # Convert bits to ASCII string
        output = get_string(bits)
        print(f"Decoded message: '{output}'")
        
    else:
        print("Data appears to be decimal values, converting to binary")
        # Convert decimals to bits (original method)
        bits = decimal_to_bits(raw_data)
        print(f"Total bits: {len(bits)}")
        print(f"First 40 bits: {bits[:40]}")
        
        # Convert bits to ASCII string
        output = get_string(bits)
        print(f"Decoded message: '{output}'")

except FileNotFoundError:
    print("File 'bits_read_in.txt' not found.")

except Exception as e:
    print(f"Error: {e}")
    
# Additional debugging - let's examine the bit pattern more closely
print("\n" + "="*50)
print("Detailed bit analysis:")

try:
    raw_data = np.fromfile("bits_read_in.txt", dtype=np.uint8)
    bits = bits_already_binary(raw_data)
    
    print(f"Total bits: {len(bits)}")
    print(f"First 8 bytes of bits: {bits[:64]}")  # Show first 8 bytes worth
    
    # Try different starting positions in case there's a header or offset
    for offset in [0, 1, 2, 3, 4, 5]:
        print(f"\nTrying offset {offset}:")
        offset_bits = bits[offset:]
        if len(offset_bits) >= 8:
            output = get_string(offset_bits)
            print(f"  First few characters: '{output[:10]}'")
            
            # Look for common ASCII patterns
            if any(32 <= ord(c) <= 126 for c in output[:20] if c):  # printable ASCII
                print(f"  Full message from offset {offset}: '{output}'")
                break

except Exception as e:
    print(f"Error in detailed analysis: {e}")