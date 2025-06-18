import numpy as np
from scipy.fft import fft, ifft

def read_qpsk(symbols):
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