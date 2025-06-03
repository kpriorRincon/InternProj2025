# TODO we need to build this class out from the file:
#sim_qpsk_noisy_demod.py

import numpy as np
class Receiver:
    def __init__(self, sampling_rate, frequency):
        #constructor
        self.sampling_rate = sampling_rate
        self.frequency = frequency
        # Phase sequence for qpsk modulation corresponds to the letter 'R'
        self.phase_start_sequence = np.array([-1+1j, -1+1j, 1+1j, 1-1j]) # this is the letter R in QPSK
        self.phases = np.array([45, 135, 225, 315])  # QPSK phase angles in degrees
    def matched_filter(self, qpsk_waveform):
        pass
    def demod():
        pass

    def get_string(self, bits):
        """Convert bits to string."""
        # Convert bits to bytes
        #take away the prefix 'R' from the bits
        bits = bits[1:]

        pass