class SigGen:
    
    def __init__(self, freq=1.0, amp=1.0, sample_rate =4e9, symbol_rate=10e6):
        import numpy as np
        self.freq = freq  # Frequency in Hz
        self.sample_rate = sample_rate  # sample rate in samples per second
        self.symbol_rate = symbol_rate  # Symbol rate in symbols per second about 30% of the frequency
        self.amp = amp    # Amplitude

        self.time_vector = None
        self.qpsk_waveform = None
        # Map bit pairs to complex symbols
        self.mapping = {
            (0, 0): (1 + 1j) / np.sqrt(2),
            (0, 1): (-1 + 1j) / np.sqrt(2),
            (1, 1): (-1 - 1j) / np.sqrt(2),
            (1, 0): (1 - 1j) / np.sqrt(2)
        }

    def generate_qpsk(self, bits):
        """
        Generate a QPSK signal from a sequence of bits.
        
        Parameters:
            bits (list): List of bits (0s and 1s) len(bits) % 2 == 0.
            sample_rate (int): Number of samples per second.
            symbol_rate (int): Number of symbols per second.
        
        Returns:
            np.ndarray: Time vector.
            np.ndarray: QPSK waveform.
            list: Vertical lines to show phase transition.
            symbols (list): List of complex symbols corresponding to the bit pairs.
        """
        import numpy as np
        
        # Convert bits to symbols
        symbols = []
        if len(bits) % 2 != 0:
            raise ValueError("Bit sequence must have an even length.")

        #seperate into odd and even and map to complex symbols
        symbols = [self.mapping[(bits[i], bits[i + 1])] for i in range(0, len(bits), 2)]

        t_vertical_lines = []  # Initialize vertical lines for debugging
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        #time vector for the wave form defined start at 0 end at the length of the symbols times samples per symbol
        # and spaced by the sample rate
        t = np.arange(0, len(symbols) * samples_per_symbol) / self.sample_rate

        # Initialize the QPSK waveform
        qpsk_waveform = np.zeros_like(t)
        #for loop that tracks the index of the symbol and the symbol itself
        for i, symbol in enumerate(symbols):
            #compute the phase offset for the symbol
            phase_offset = np.angle(symbol)
            #debugging print statement to show the symbol and phase offset
            #print(f"Symbol {i}: {symbol}, Phase Offset: {phase_offset}");
            idx_start = i * samples_per_symbol
            idx_end = idx_start + samples_per_symbol
            time_slice = t[idx_start:idx_end] # time slice for the current symbol
            qpsk_waveform[idx_start:idx_end] = (
                #eqn from Haupt pg 86
                1/np.sqrt(2)*np.cos(2*self.freq*np.pi * time_slice + phase_offset)
                #add vertical dashed lines at time slices of the symbols
            )
            t_vertical_lines.append(idx_start/self.sample_rate)

        self.time_vector = t
        self.qpsk_waveform = qpsk_waveform
        return t, qpsk_waveform, t_vertical_lines, symbols
    
    def message_to_bits(self, message):
        """
        Author: Skylar Harris
        Convert a string message to a list of bits.
        
        Parameters:
            message (str): The input string message.
        
        Returns:
            list: List of bits (0s and 1s).
        """
        # prefix the message with the letter 'R' as a marker
        message = 'R' + message
        message_binary = ''.join(format(ord(x), '08b') for x in message)
        # print(message_binary)
        # Convert string input to list of integers
        bit_sequence = [int(bit) for bit in message_binary.strip()]
        return bit_sequence
