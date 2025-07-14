import numpy as np

start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                1, 0, 1, 0, 0, 1, 0, 0,
                0, 0, 1, 0, 1, 0, 1, 1,
                1, 0, 1, 1, 0, 0, 0, 1]

def generate_qpsk(bits):
        """ 
        Convert bits to QPSK Symbols 
        Mapping used is for grey coding
        Taken and simplified from the 
        Sig Gen class used in simulation

        """
        mapping = {
            (1, 1): (1 + 1j) / np.sqrt(2),
            (0, 1): (-1 + 1j) / np.sqrt(2),
            (0, 0): (-1 - 1j) / np.sqrt(2),
            (1, 0): (1 - 1j) / np.sqrt(2)
        }

        # add the start sequence to the bits
        bits_to_send = start_sequence + bits
        
        # Convert bits to symbols
        if len(bits_to_send) % 2 != 0:
            raise ValueError("Bit sequence must have an even length.")
        
        # Map bit pairs to complex symbols
        symbols = [mapping[(bits_to_send[i], bits_to_send[i + 1])] for i in range(0, len(bits_to_send), 2)]

        return symbols