import numpy as np

def get_string_from_bytes(byte_data):
    """Convert byte data directly to ASCII string"""
    ascii_chars = []
    
    for byte_val in byte_data:
        # Handle signed/unsigned conversion if needed
        if byte_val < 0:
            byte_val = byte_val + 256
            
        # Make ASCII characters (only printable ones)
        if 32 <= byte_val <= 126:  # Printable ASCII range
            ascii_chars.append(chr(byte_val))
        elif byte_val == 0:
            ascii_chars.append('\\0')  # Show null characters
        else:
            ascii_chars.append(f'[{byte_val}]')  # Show non-printable as [value]

    return ''.join(ascii_chars)

def bytes_to_bits(byte_data):
    """Convert bytes to individual bits for analysis"""
    all_bits = []
    for byte_val in byte_data:
        # Handle signed values
        if byte_val < 0:
            byte_val = byte_val + 256
        # Convert to 8-bit binary
        bits = [(byte_val >> i) & 1 for i in range(7, -1, -1)]  # MSB first
        all_bits.extend(bits)
    return all_bits

def bits_to_string(bits):
    """Convert bit array to ASCII string"""
    ascii_chars = []
    
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):
            byte_bits = bits[i:i+8]
            byte_val = 0
            for j, bit in enumerate(byte_bits):
                byte_val |= (bit << (7 - j))  # MSB first
            
            if 32 <= byte_val <= 126:
                ascii_chars.append(chr(byte_val))
            elif byte_val == 0:
                ascii_chars.append('\\0')
            else:
                ascii_chars.append(f'[{byte_val}]')
    
    return ''.join(ascii_chars)

def analyze_gnu_radio_output():
    """Analyze GNU Radio output file with different data types"""
    
    data_types = [
        ('uint8', np.uint8, "Unsigned 8-bit (0-255)"),
        ('int8', np.int8, "Signed 8-bit (-128 to 127)"),
        ('int16', np.int16, "Signed 16-bit"),
        ('int32', np.int32, "Signed 32-bit"), 
        ('float32', np.float32, "32-bit float"),
        ('complex64', np.complex64, "Complex 64-bit (32-bit real + 32-bit imag)")
    ]
    
    print("Analyzing 'bits_read_in.txt' with different GNU Radio data types:")
    print("=" * 70)
    
    for dtype_name, numpy_dtype, description in data_types:
        try:
            print(f"\n{description} ({dtype_name}):")
            data = np.fromfile("bits_read_in.txt", dtype=numpy_dtype)
            print(f"  File contains {len(data)} samples")
            
            if len(data) == 0:
                print("  No data read!")
                continue
                
            print(f"  First 10 values: {data[:10]}")
            print(f"  Value range: {np.min(data)} to {np.max(data)}")
            
            if dtype_name in ['uint8', 'int8']:
                # These are likely the actual decoded bytes
                decoded = get_string_from_bytes(data)
                print(f"  Direct ASCII: '{decoded[:50]}{'...' if len(decoded) > 50 else ''}'")
                
                # Also try as individual bits (in case unpack_k_bits outputs bit values)
                if all(val in [0, 1] for val in data):
                    bit_decoded = bits_to_string(data)
                    print(f"  As bits: '{bit_decoded[:50]}{'...' if len(bit_decoded) > 50 else ''}'")
                
            elif dtype_name in ['int16', 'int32']:
                # Convert to bytes first
                if dtype_name == 'int16':
                    # Each int16 might contain 2 bytes
                    bytes_data = []
                    for val in data:
                        if val < 0:
                            val = val + (1 << 16)  # Convert to unsigned
                        bytes_data.append(val & 0xFF)  # Low byte
                        bytes_data.append((val >> 8) & 0xFF)  # High byte
                else:  # int32
                    # Each int32 might contain 4 bytes
                    bytes_data = []
                    for val in data:
                        if val < 0:
                            val = val + (1 << 32)  # Convert to unsigned
                        for i in range(4):
                            bytes_data.append((val >> (i * 8)) & 0xFF)
                
                decoded = get_string_from_bytes(bytes_data)
                print(f"  As bytes: '{decoded[:50]}{'...' if len(decoded) > 50 else ''}'")
                
            elif dtype_name == 'float32':
                # Float values might represent soft bits or normalized data
                print(f"  Unique values: {len(np.unique(data))}")
                if len(np.unique(data)) <= 10:  # Likely discrete values
                    print(f"  All unique values: {np.unique(data)}")
                    
                # Try thresholding if they look like soft bits
                if np.all(data >= -2) and np.all(data <= 2):
                    thresholded = (data > 0).astype(int)
                    if len(thresholded) % 8 == 0:
                        bit_decoded = bits_to_string(thresholded)
                        print(f"  Thresholded bits: '{bit_decoded[:50]}{'...' if len(bit_decoded) > 50 else ''}'")
                
            elif dtype_name == 'complex64':
                print(f"  Real part range: {np.min(data.real)} to {np.max(data.real)}")
                print(f"  Imag part range: {np.min(data.imag)} to {np.max(data.imag)}")
                # This would be the modulated signal, not the demodulated bits
                
        except Exception as e:
            print(f"  Error reading as {dtype_name}: {e}")
    
    # Special analysis for the most likely case (uint8/int8)
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS - Most likely formats:")
    
    for dtype_name, numpy_dtype in [('uint8', np.uint8), ('int8', np.int8)]:
        try:
            data = np.fromfile("bits_read_in.txt", dtype=numpy_dtype)
            if len(data) > 0:
                print(f"\n{dtype_name.upper()} DETAILED:")
                decoded = get_string_from_bytes(data)
                print(f"Full message: '{decoded}'")
                
                # Show hex dump style output
                print("Hex dump (first 64 bytes):")
                for i in range(0, min(64, len(data)), 16):
                    hex_vals = ' '.join(f'{b:02X}' if b >= 0 else f'{b+256:02X}' 
                                      for b in data[i:i+16])
                    ascii_vals = ''.join(chr(b) if 32 <= b <= 126 else '.' 
                                       for b in data[i:i+16] 
                                       for b in [b if b >= 0 else b + 256])
                    print(f"  {i:04X}: {hex_vals:<48} {ascii_vals}")
                    
        except Exception as e:
            print(f"Error in detailed analysis for {dtype_name}: {e}")

if __name__ == "__main__":
    analyze_gnu_radio_output()