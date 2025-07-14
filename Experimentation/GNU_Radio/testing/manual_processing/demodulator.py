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

    # print("Reading bits from symbols")
    bits = np.zeros((len(symbols), 2), dtype=int)
    for i in range(len(symbols)):
        angle = np.angle(symbols[i], deg=True) % 360

        # codex for the phases to bits
        if 0 <= angle < 90:
            bits[i] = [1, 1]  # 45°
        elif 90 <= angle < 180:
            bits[i] = [0, 1]  # 135°
        elif 180 <= angle < 270:
            bits[i] = [0, 0]  # 225°
        else:
            bits[i] = [1, 0]  # 315°
    
    # put into a single list
    best_bits = ''.join(str(b) for pair in bits for b in pair)
    return best_bits

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