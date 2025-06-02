# TODO we need to build this class out from the file:
#sim_qpsk_noisy_demod.py

import numpy as np
class Receiver:
    def __init__(self, sampling_rate, frequency):
        self.sampling_rate = sampling_rate
        self.frequency = frequency
        self.phase_start_sequence = np.array([-1+1j, -1+1j, 1+1j, 1-1j]) # this is the letter R in QPSK
        self.phases = np.array([45, 135, 225, 315])  # QPSK phase angles in degrees


    def bit_reader(symbols):
        """
        Convert QPSK symbols to bits.
        
        Parameters:
            symbols (np.ndarray): Array of QPSK symbols.
        
        Returns:
            np.ndarray: Array of bits corresponding to the QPSK symbols.
        """
        bits = np.zeros((len(symbols), 2), dtype=int)
        for i in range(len(symbols)):
            angle = np.angle(symbols[i], deg=True) % 360

            # don't know why, but this is the only way to get the angles in the right range.
            # Might be because of the Hilbert transform?
            if 0 <= angle < 90:
                bits[i] = [0, 0]  # 45°
            elif 90 <= angle < 180:
                bits[i] = [0, 1]  # 135°
            elif 180 <= angle < 270:
                bits[i] = [1, 1]  # 225°
            else:
                bits[i] = [1, 0]  # 315°
            return bits
        
    def decoder(self, qpsk_waveform, sample_rate, symbol_rate):
        """
        Decode the QPSK waveform to retrieve the original bits.
        
        Parameters:
            qpsk_waveform (np.ndarray): The QPSK waveform to decode.
            sample_rate (int): The sampling rate of the waveform.
            symbol_rate (int): The symbol rate of the waveform.
        
        Returns:
            np.ndarray: The decoded bits.
        """
        from scipy.signal import hilbert
        
        # Compute the Hilbert transform
        analytic_signal = hilbert(qpsk_waveform)
        ## Sample at symbol midpoints ##
        samples_per_symbol = int(sample_rate / symbol_rate)             # number of samples per symbol
        offset = samples_per_symbol // 2                                # offset to sample at the midpoint of each symbol   
        sampled_symbols = analytic_signal[offset::samples_per_symbol]   # symbols sampled from the analytical signal
        sampled_symbols /= np.abs(sampled_symbols)                      # normalize the symbols

        ## look for the start sequence ##
        expected_start_sequence = ''.join(str(bit) for pair in self.bit_reader(self.phase_start_sequence) for bit in pair)    # put the start sequence into a string
        best_bits = None                                                                                            # holds the best bits found
        #print("Expected Start Sequence: ", expected_start_sequence)                                                 # debug statement
        og_sampled_symbols = ''.join(str(bit) for pair in self.bit_reader(sampled_symbols) for bit in pair)              # original sampled symbols in string format
        #print("Original sampled bits: ", og_sampled_symbols)                                                        # debug statement

        ## Loop through possible phase shifts ##
        for i in range(0, 3):   # one for each quadrant (0°, 90°, 180°, 270°)
            # Rotate the flat bits to match the start sequence
            rotated_bits = sampled_symbols * np.exp(-1j* np.deg2rad(i*90))  # Rotate by 0, 90, 180, or 270 degrees
            
            # decode the bits
            decode_bits = self.bit_reader(rotated_bits)                                  # decode the rotated bits
            flat_bits = ''.join(str(bit) for pair in decode_bits for bit in pair)   # put the bits into a string
            #print("Rotated bits: ", flat_bits)                                      # debug statement
            
            # Check for presence of the known start sequence (first few symbols)
            if expected_start_sequence == flat_bits[0:8]:                   # check only first 8 symbols worth (16 bits)
                #print(f"Start sequence found with phase shift: {i*90}°")
                best_bits = flat_bits                                       # store the best bits found
                break
        
        # Error state if no start sequence was found
        if best_bits is None:
            print("Start sequence not found. Defaulting to 0°")
            rotated_symbols = sampled_symbols
            decoded_bits = self.bit_reader(rotated_symbols)
            best_bits = ''.join(str(b) for pair in decoded_bits for b in pair)

        return analytic_signal, best_bits
