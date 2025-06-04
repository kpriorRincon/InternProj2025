# TODO we need to build this class out from the file:
#sim_qpsk_noisy_demod.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

class Receiver:
    def __init__(self, sampling_rate, frequency):
        #constructor
        self.sampling_rate = sampling_rate
        self.frequency = frequency
        # Phase sequence for qpsk modulation corresponds to the letter 'R'
        self.phase_start_sequence = np.array([-1+1j, -1+1j, 1+1j, 1-1j]) # this is the letter R in QPSK
        self.phases = np.array([45, 135, 225, 315])  # QPSK phase angles in degrees
   
    # Sample the received signal
    def sample_signal(self, analytic_signal, sample_rate, symbol_rate):
        print("Sampling the analytic signal")
        ## Sample at symbol midpoints ##
        samples_per_symbol = int(sample_rate / symbol_rate)             # number of samples per symbol
        offset = samples_per_symbol // 2                                # offset to sample at the midpoint of each symbol   
        ## Sample the analytic signal ##
        sampled_symbols = analytic_signal[offset::samples_per_symbol]   # symbols sampled from the analytical signal
        sampled_symbols /= np.abs(sampled_symbols)                      # normalize the symbols
        print("Done sampling the analytic signal")
        return sampled_symbols
    
    # QPSK symbol to bit mapping
    def bit_reader(self, symbols):
        print("Reading bits from symbols")
        bits = np.zeros((len(symbols), 2), dtype=int)
        for i in range(len(symbols)):
            angle = np.angle(symbols[i], deg=True) % 360

            # codex for the phases to bits
            if 0 <= angle < 90:
                bits[i] = [0, 0]  # 45°
            elif 90 <= angle < 180:
                bits[i] = [0, 1]  # 135°
            elif 180 <= angle < 270:
                bits[i] = [1, 1]  # 225°
            else:
                bits[i] = [1, 0]  # 315°
        return bits

    # Error checking for the start sequence
    def matched_filter(self, sampled_symbols):
        print("Error checking")
        ## look for the start sequence ##
        expected_start_sequence = ''.join(str(bit) for pair in self.bit_reader(self.phase_start_sequence) for bit in pair)  # put the start sequence into a string
        best_bits = None                                                                                                    # holds the best bits found
        print("Expected Start Sequence: ", expected_start_sequence)                                                         # debug statement
        og_sampled_symbols = ''.join(str(bit) for pair in self.bit_reader(sampled_symbols) for bit in pair)                 # original sampled symbols in string format
        print("Sampled bits: ", og_sampled_symbols)                                                                         # debug statement

        ## Loop through possible phase shifts ##
        for i in range(0, 3):   # one for each quadrant (0°, 90°, 180°, 270°)
            # Rotate the flat bits to match the start sequence
            rotated_bits = sampled_symbols * np.exp(-1j* np.deg2rad(i*90))  # Rotate by 0, 90, 180, or 270 degrees
            
            # decode the bits
            decode_bits = self.bit_reader(rotated_bits)                             # decode the rotated bits
            flat_bits = ''.join(str(bit) for pair in decode_bits for bit in pair)   # put the bits into a string
            print("Rotated bits: ", flat_bits)                                      # debug statement
            
            # Check for presence of the known start sequence (first few symbols)
            if expected_start_sequence == flat_bits[0:8]:                   # check only first 8 symbols worth (16 bits)
                print(f"Start sequence found with phase shift: {i*90}°")
                best_bits = flat_bits                                       # store the best bits found
                break
        
        # Error state if no start sequence was found
        if best_bits is None:
            print("Start sequence not found. Defaulting to 0°")
            rotated_symbols = sampled_symbols
            decoded_bits = self.bit_reader(rotated_symbols)
            best_bits = ''.join(str(b) for pair in decoded_bits for b in pair)
        
        return best_bits
    

    # sample the received signal and do error checking
    def demodulator(self, qpsk_waveform, sample_rate, symbol_rate):
        ## Apply matched filter to the received signal ##
        #correlated_signal = self.matched_filter(qpsk_waveform, f_base_band, symbol_rate, N)  # apply the matched filter to the received signal
        correlated_signal = qpsk_waveform

        ## compute the Hilbert transform ##
        print("Applying Hilbert Transform...")
        analytic_signal = hilbert(np.real(correlated_signal))  # hilbert transformation
        
        # sample the analytic signal
        print("Sampling the analytic signal...")
        sampled_symbols = self.sample_signal(analytic_signal, sample_rate, symbol_rate)

        # decode the symbols and error check the start sequence
        print("Decoding symbols and checking for start sequence...")
        best_bits = self.matched_filter(sampled_symbols)

        return analytic_signal, best_bits

    def get_string(self, bits):
        """Convert bits to string."""
        # Convert bits to bytes
        #take away the prefix 'R' from the bits
        bits = bits[1:]

        pass