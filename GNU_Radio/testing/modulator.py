import numpy as np
from scipy.fft import fft, ifft

def generate_qpsk(bits):
        mapping = {
            (1, 1): (1 + 1j) / np.sqrt(2),
            (0, 1): (-1 + 1j) / np.sqrt(2),
            (0, 0): (-1 - 1j) / np.sqrt(2),
            (1, 0): (1 - 1j) / np.sqrt(2)
        }
        
        # Convert bits to symbols
        if len(bits) % 2 != 0:
            raise ValueError("Bit sequence must have an even length.")
        
        # Map bit pairs to complex symbols
        symbols = [mapping[(bits[i], bits[i + 1])] for i in range(0, len(bits), 2)]
        
        return symbols

def rrc_filter(beta, N, Ts, fs):
        # The number of samples in each symbol
        samples_per_symbol = int(fs * Ts)

        # The filter span in symbols
        total_symbols = N / samples_per_symbol

        # The total amount of time that the filter spans
        total_time = total_symbols * Ts

        # The time vector to compute the impulse response
        time = np.linspace(-total_time / 2, total_time / 2, N, endpoint=False)

        # ---------------------------- Generating the RRC impulse respose ----------------------------

        # The root raised-cosine impulse response is generated from taking the square root of the raised-cosine impulse response in the frequency domain

        # Raised-cosine filter impulse response in the time domain
        num = np.cos( (np.pi * beta * time) / (Ts) )
        denom = 1 - ( (2 * beta * time) / (Ts) ) ** 2
        g = np.sinc(time / Ts) * (num / denom)

        # Raised-cosine filter impulse response in the frequency domain
        fg = fft(g)

        # Root raised-cosine filter impulse response in the frequency domain
        fh = np.sqrt(fg)

        # Root raised-cosine filter impulse respone in the time domain
        h = ifft(fh)

        return h 