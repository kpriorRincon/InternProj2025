import numpy as np

class transmit_processing:

    def __init__(self, sps, sample_rate):
         self.sps = sps
         self.sample_rate = sample_rate

    # function that converts a message to its bit sequence
    def message_to_bits(self, message):
        """
        Convert a string message to a bitstream 

        Parameters:
        - message (str) : String message

        Returns:
        - bit_sequence : Bit sequence as a list
        """
        start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                            1, 0, 1, 0, 0, 1, 0, 0,
                            0, 0, 1, 0, 1, 0, 1, 1,
                            1, 0, 1, 1, 0, 0, 0, 1]

        end_sequence = [0, 0, 1, 0, 0, 1, 1, 0,
                        1, 0, 0, 0, 0, 0, 1, 0,
                        0, 0, 1, 1, 1, 1, 0, 1,
                        0, 0, 0, 1, 0, 0, 1, 0]

        # convert message to binary
        message_binary = ''.join(format(ord(char), '08b') for char in message)

        # add start and end markers to the bitstream
        message_binary = ''.join(str(bit) for bit in start_sequence) + message_binary + ''.join(str(bit) for bit in end_sequence)
    
        # coverts bit sequence from a string to a list of integers
        bit_sequence = [int(bit) for bit in message_binary.strip()]

        return bit_sequence
    
    # function to map bits to QPSK symbols
    def qpsk_mapping(self, bits):
        """
        Map a bitstream to QPSK complex symbols

        Parameters:
        - bits : Bit sequence to be mapped

        Returns:
        - symbols : QPSK mapped symbols 
        """
        num_bits = len(bits)
        num_symbols = num_bits // 2

        mapping =(1/np.sqrt(2)) * np.array([
             1 + 1j,  # 00
            -1 + 1j,  # 01
             1 - 1j,  # 10
            -1 - 1j   # 11
        ], dtype=np.complex64)

        symbols = np.zeros(num_symbols, dtype=np.complex64)

        for i in range(num_symbols):
            bit1 = bits[2 * i]
            bit2 = bits[2 * i + 1]
            symbol_index = bit1 * 2**1 + bit2 * 2**0
            symbols[i] = mapping[symbol_index]
        
        return symbols

    # function to upsample the symbols
    def insert_zeros(self, sps, symbols):
        """
        Insert zeros in between symbols to upsample to desired sps

        Parameters:
        - sps : Desired samples per symbol

        Returns:
        - upsampled_symbols : Zero-padded symbols
        """
        num_symbols = len(symbols)
        upsampled_symbols = np.zeros(num_symbols * sps, dtype=np.complex64)
        upsampled_symbols[::sps] = symbols

        return upsampled_symbols

    # root raised cosine filter generation function
    def rrc_filter(self, beta, N, Ts, fs):
        """
        Generate a Root Raised-Cosine (RRC) filter (FIR) impulse response

        Parameters:
        - beta : Roll-off factor (0 < beta <= 1)
        - N : Total number of taps in the filter (the filter span)
        - Ts : Symbol period 
        - fs : Sampling frequency/rate (Hz)

        Returns:
        - time : The time vector of the impulse response
        - h : The impulse response of the RRC filter in the time domain
        """
        t = np.arange(-N // 2, N // 2 + 1) / fs 

        h = np.zeros_like(t) 

        for i in range(len(t)):
            if t[i] == 0.0:
                h[i] = (1.0 + beta * (4/np.pi - 1))
            elif abs(t[i]) == Ts / (4 * beta):
                h[i] = (beta / np.sqrt(2)) * (
                    ((1 + 2/np.pi) * np.sin(np.pi / (4 * beta))) +
                    ((1 - 2/np.pi) * np.cos(np.pi / (4 * beta)))
                )
            else:
                numerator = np.sin(np.pi * t[i] * (1 - beta) / Ts) + 4 * beta * t[i] / Ts * np.cos(np.pi * t[i] * (1 + beta) / Ts)
                denominator = np.pi * t[i] * (1 - (4 * beta * t[i] / Ts) ** 2) / Ts
                h[i] = numerator / denominator
        return t, h
    
    def work(self, message, beta, N): 
        """
        Execute all transmit processing

        Parameters:
        - message : String message
        - beta : Roll-off factor (0 < beta <= 1)
        - N : Total number of taps in the filter (the filter span)

        Returns:
        - bits : Modulated bits
        - data : IQ data to be sent to transmitter
        """

        bits = self.message_to_bits(message)

        bits_string = ''.join(str(b) for b in bits)

        symbols = self.qpsk_mapping(bits)

        upsampled_symbols = self.insert_zeros(self.sps, symbols)

        symbol_rate = self.sample_rate / self.sps
        Ts = 1 / symbol_rate

        _, h = self.rrc_filter(beta, N, Ts, self.sample_rate)
        data = np.convolve(upsampled_symbols, h, 'same')
        
        return bits_string, data
