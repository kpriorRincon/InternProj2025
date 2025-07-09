import numpy as np
from scipy.signal import fftconvolve,  max_len_seq
from crc import Calculator, Crc8

class transmit_processing:
    
    def __init__(self, sps, sample_rate):
        self.sps = sps
        self.sample_rate = sample_rate
        self.start_sequence = [1, 1, 1, 1, 1, 1, 1, 0,
                               0, 0, 1, 1, 1, 0, 1, 1,
                               0, 0, 0, 1, 0, 1, 0, 0,
                               1, 0, 1, 1, 1, 1, 1, 0,
                               1, 0, 1, 0, 1, 0, 0, 0,
                               0, 1, 0, 1, 1, 0, 1, 1,
                               1, 1, 0, 0, 1, 1, 1, 0,
                               0, 1, 0, 1, 0, 1, 1, 0,
                               0, 1, 1, 0, 0, 0, 0, 0,
                               1, 1, 0, 1, 1, 0, 1, 0,
                               1, 1, 1, 0, 1, 0, 0, 0,
                               1, 1, 0, 0, 1, 0, 0, 0,
                               1, 0, 0, 0, 0, 0, 0, 1,
                               0, 0, 1, 0, 0, 1, 1, 0,
                               1, 0, 0, 1, 1, 1, 1, 0,
                               1, 1, 1, 0, 0, 0, 0, 1 ]

        self.end_sequence = [1, 1, 1, 0, 0, 0, 0, 1,
                             1, 1, 1, 0, 0, 1, 1, 0,
                             1, 0, 1, 0, 1, 0, 0, 1,
                             1, 0, 1, 0, 1, 0, 0, 0,
                             1, 1, 1, 1, 0, 1, 1, 1,
                             0, 1, 0, 0, 1, 0, 1, 1,
                             1, 1, 1, 1, 1, 1, 0, 1,
                             0, 0, 1, 1, 0, 1, 0, 1,
                             1, 1, 1, 1, 1, 1, 0, 1,
                             1, 0, 1, 0, 1, 0, 1, 0,
                             0, 1, 1, 1, 0, 0, 0, 0,
                             1, 1, 1, 0, 0, 0, 1, 0,
                             0, 1, 0, 1, 0, 0, 1, 1,
                             0, 1, 1, 1, 0, 1, 0, 1,
                             0, 1, 0, 1, 0, 1, 1, 0,
                             0, 0, 1, 1, 0, 1, 0, 1]
            
    def generate_markers(self):
        """
        Generates start and end markers for the signal

        Returns:
        - start_sequence : start marker
        - end_sequence : end marker
        """

        random_sequence = max_len_seq(11)[0]
        idx = (len(random_sequence) - 1) // 2
        start_sequence = random_sequence[:idx]
        end_sequence = random_sequence[idx:-1]

        return start_sequence, end_sequence
        
    # function that converts a message to its bit sequence
    def message_to_bits(self, message):
        """
        Convert a string message to a bitstream 

        Parameters:
        - message (str) : String message

        Returns:
        - bit_sequence : Bit sequence as a list
        """
 
        # convert message to binary
        message_binary = ''.join(format(ord(char), '08b') for char in message)

        # add start and end markers to the bitstream
        message_binary = ''.join(str(bit) for bit in self.start_sequence) + message_binary + ''.join(str(bit) for bit in self.end_sequence)
    
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

        t = np.linspace(-N//2, N//2, N) / fs 

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
        return t, h/np.sqrt(np.sum(h**2))  # Normalize to get unity gain
    
    # function that modulates the start and end markers of the signal 
    def modulated_markers(self, beta, N):
        """
        Modulate start and end sequences

        Parameters:
        - beta : Roll-off factor (0 < beta <= 1)
        - N : Total number of taps in the filter (the filter span)

        Returns:
        - start_data : Modulated start sequence
        - end_data : Modulated end sequence
        """ 
     
        start_sequence = ''.join(str(bit) for bit in self.start_sequence)
        end_sequence =  ''.join(str(bit) for bit in self.end_sequence)

        start_sequence = [int(bit) for bit in start_sequence.strip()]
        end_sequence = [int(bit) for bit in end_sequence.strip()]

        start_symbols = self.qpsk_mapping(start_sequence)
        end_symbols = self.qpsk_mapping(end_sequence)

        upsampled_start_symbols = self.insert_zeros(self.sps, start_symbols)
        upsampled_end_symbols = self.insert_zeros(self.sps, end_symbols)
		
        symbol_rate = self.sample_rate / self.sps
        Ts = 1 / symbol_rate

        _, h = self.rrc_filter(beta, N, Ts, self.sample_rate)
        start_data = fftconvolve(upsampled_start_symbols, h, 'same')
        start_data = start_data.astype(np.complex64)
        end_data = fftconvolve(upsampled_end_symbols, h, 'same')
        end_data = end_data.astype(np.complex64)

        return start_data, end_data

    def add_crc(self, message):
        """
        Add CRC-8 to the message
        
        Parameters:
        - message: String message to which CRC-8 will be added
        
        Returns:
        - to_send: Byte data with CRC-8 appended
        """
        
        # user input to bytes
        byte_data = bytes(message, "ascii")
        print("Message in bytes: ", byte_data)

        # calculate CRC-8 for the message
        calculator = Calculator(Crc8.CCITT)
        crc_code = calculator.checksum(byte_data)
        print("CRC-8: ", crc_code)

        # append the crc-8 to the message
        to_send = byte_data + bytes([crc_code])
        print("Data with CRC-8 appended: ", to_send)

        # turn into a bit string
        bit_string = ''.join(format(byte, '08b') for byte in to_send)

        return bit_string
    
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
        # start_sequence, end_sequence = self.generate_markers()

        # add CRC to the message
        bits_string = self.add_crc(message)
        bits = self.message_to_bits(bits_string)

        # map to QPSK symbols
        symbols = self.qpsk_mapping(bits)

        # upsample the symbols
        upsampled_symbols = self.insert_zeros(self.sps, symbols)

        # pulse shaping using the RRC filter
        symbol_rate = self.sample_rate / self.sps
        Ts = 1 / symbol_rate
        _, h = self.rrc_filter(beta, N, Ts, self.sample_rate)
        data = np.convolve(upsampled_symbols, h, 'same')
        data = data.astype(np.complex64)

        # append zeros so there is space between packets
        zeros = np.zeros(len(data)*3, dtype=np.complex64)
        data = np.append(data, zeros)
        
        # save data to a file for testing
        data.tofile("data_for_sighound.bin")
        
        return bits_string, data		
