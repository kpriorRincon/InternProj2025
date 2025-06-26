import numpy as np

class receive_processing:

    def __init__(self, sps, sample_rate):
         self.sps = sps
         self.sample_rate = sample_rate

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
    
    # function to decimate the data
    def decimate(self, sps, data):
        """
        Decimate a complex signal with sps > 1 to one with sps = 1

        Parameters:
        - sps : Samples per symbol of data
        - data : Data to be decimated
        
        Returns:
        - symbols : Decimated data - the symbols to be demodulated
        """
        symbols = data[::sps]

        return symbols
    
    # function to map the complex symbols to bits
    def qpsk_demodulator(self, symbols):
        """
        Map a QPSK complex symbols back to bits

        Parameters:
        - symbols : QPSK mapped symbols

        Returns:
        - bits : Demodulated bit sequence 
        """
        num_symbols = len(symbols)
        num_bits = num_symbols * 2

        bits = np.zeros(num_bits)

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

        bits = bits.astype(int).tolist()

        return bits
    
    # function that converts a bitstream to a message
    def bits_to_message(self, bits):
        """
        Convert a bitstream to a string message

        Parameters:
        - bits : Bit sequence

        Returns:
        - message (str) : Decoded message
        """
        bits = bits[32:-32] # Take out markers

        bits_string = ''.join(str(bit) for bit in bits)

        message = ''.join(chr(int(bits_string[i*8:i*8+8],2)) for i in range(len(bits)//8))

        return message
    
    def work(self, data, beta, N): 
        """
        Execute all transmit processing

        Parameters:
        - data : Time, phase, and frequency corrected data
        - beta : Roll-off factor (0 < beta <= 1)
        - N : Total number of taps in the filter (the filter span)

        Returns:
        - bits : Demodulated bits
        - message : Decoded message
        """
        symbol_rate = self.sample_rate / self.sps
        Ts = 1 / symbol_rate

        _, h = self.rrc_filter(beta, N, Ts, self.sample_rate)
        rc_filtered_data = np.convolve(data, h, 'same')

        symbols = self.decimate(self.sps, rc_filtered_data)

        bits = self.qpsk_demodulator(symbols)

        bits_string = ''.join(str(b) for b in bits)

        message = self.bits_to_message(bits)
        
        return bits_string, message
