class SigGen:
    def __init__(self, freq=1.0, amp=1.0):
        self.freq = freq  # Frequency in Hz
        self.amp = amp    # Amplitude

    def generate_qpsk(self, bits, sample_rate, symbol_rate):
        """
        Generate a QPSK signal from a sequence of bits.
        
        Parameters:
            bits (list): List of bits (0s and 1s) len(bits) % 2 == 0.
            sample_rate (int): Number of samples per second.
            symbol_rate (int): Number of symbols per second.
        
        Returns:
            np.ndarray: Time vector.
            np.ndarray: QPSK waveform.
        """
        import numpy as np
        
        # Convert bits to symbols
        symbols = []
        if len(bits) % 2 != 0:
            raise ValueError("Bit sequence must have an even length.")
         # Map bit pairs to complex symbols
        mapping = {
            (0, 0): (1 + 1j) / np.sqrt(2),
            (0, 1): (-1 + 1j) / np.sqrt(2),
            (1, 1): (-1 - 1j) / np.sqrt(2),
            (1, 0): (1 - 1j) / np.sqrt(2)
        }
        #seperate into odd and even and map to complex symbols
        symbols = [mapping[(bits[i], bits[i + 1])] for i in range(0, len(bits), 2)]

        samples_per_symbol = int(sample_rate / symbol_rate)
        #time vector for the wave form defined start at 0 end at the length of the symbols times samples per symbol
        # and spaced by the sample rate
        t = np.arange(0, len(symbols) * samples_per_symbol) / sample_rate

        # Initialize the QPSK waveform
        qpsk_waveform = np.zeros_like(t)
        #for loop that tracks the index of the symbol and the symbol itself
        for i, symbol in enumerate(symbols):
            #compute the phase offset for the symbol
            phase_offset = np.angle(symbol)
            idx_start = i * samples_per_symbol
            idx_end = idx_start + samples_per_symbol
            time_slice = t[idx_start:idx_end] # time slice for the current symbol
            qpsk_waveform[idx_start:idx_end] = (
                #eqn from Haupt pg 86
                np.sqrt(2)*np.cos(2*self.freq*np.pi * time_slice + phase_offset)
            )
        return t, qpsk_waveform