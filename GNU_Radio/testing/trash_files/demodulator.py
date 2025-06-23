import numpy as np
from scipy.fft import fft, ifft

# Gold start sequence to add for signal detection
start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                1, 0, 1, 0, 0, 1, 0, 0,
                0, 0, 1, 0, 1, 0, 1, 1,
                1, 0, 1, 1, 0, 0, 0, 1]

def read_qpsk(symbols):
    """ 
    Convert QPSK symbols to bits
    Mapping used is for grey coding
    Taken from the Receiver class 
    used in simulation
    
    """

    symbols = input_items[0]
    bits = output_items[0]

    num_symbols = len(symbols)
    num_bits = len(symbols) * 2

    for i in range(num_symbols):
        complex_number = symbols[i]
        real = np.real(complex_number)
        imag = np.imag(complex_number)

        # Determine bits based on quadrant
        if real > 0 and imag > 0:
            bit1 = 0
            bit2 = 0
        elif real < 0 and imag > 0:
            bit1 = 0
            bit2 = 1
        elif real > 0 and imag < 0:
            bit1 = 1
            bit2 = 0
        elif real < 0 and imag < 0:
            bit1 = 1
            bit2 = 1

        bits[2 * i] = bit1
        bits[2 * i + 1] = bit2

    return len(bits)

def phase_rotation_handler(sampled_symbols):
    """ 
    Rotates the sampled symbols to find the best phase alignment
    to match the start sequence.
    Taken from the Receiver class used in simulation
    
    """
    ## look for the start sequence ##
    expected_start_sequence = ''.join(str(bit) for bit in start_sequence) # put the start sequence into a string
    best_bits = None                                                                                                                                                    # debug statement

    ## Loop through possible phase shifts ##
    for i in range(0, 7):   # one for each quadrant (0°, 90°, 180°, 270°)
        # rotation amount
        rotation = i * 45
        # Rotate the flat bits to match the start sequence
        rotated_bits = sampled_symbols * np.exp(-1j* np.deg2rad(rotation))  # Rotate by multiples of 45 degrees
        # decode the bits
        decoded_bits = read_qpsk(rotated_bits)                             # decode the rotated bits
        # Check for presence of the known start sequence (first few symbols)
        if expected_start_sequence == decoded_bits[0:32]:                  # check only first 16 symbols worth (32 bits)
            print("Found start sequence with rotation: ", rotation, "degrees")
            best_bits = decoded_bits                                       # store the best bits found
            break
    
    # Error state if no start sequence was found
    if best_bits is None:
        print("Start sequence not found... \nUsing default samples.")
        rotated_symbols = sampled_symbols
        best_bits = read_qpsk(rotated_symbols)
    
    return best_bits