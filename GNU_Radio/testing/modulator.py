import numpy as np

def generate_qpsk(bits):
        mapping = {
            (0, 0): (1 + 1j) / np.sqrt(2),
            (0, 1): (-1 + 1j) / np.sqrt(2),
            (1, 1): (-1 - 1j) / np.sqrt(2),
            (1, 0): (1 - 1j) / np.sqrt(2)
        }
        
        # Convert bits to symbols
        if len(bits) % 2 != 0:
            raise ValueError("Bit sequence must have an even length.")
        
        # Map bit pairs to complex symbols
        symbols = [mapping[(bits[i], bits[i + 1])] for i in range(0, len(bits), 2)]
        
        return symbols