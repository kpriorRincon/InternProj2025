import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

def rrc_filter(beta, N, Ts, fs):
    
    """
    Generate a Root Raised-Cosine (RRC) filter (FIR) impulse response

    Parameters:
    - beta : Roll-off factor (0 < beta <= 1)
    - N : Total number of taps in the filter (the filter span)
    - Ts : Symbol period 
    - fs : Sampling frequency/rate (Hz)

    Returns:
    - h : The impulse response of the RRC filter in the time domain
    - time : The time vector of the impulse response

    """

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

    return time, h 

# Testing of the filter
symbol_rate = 500
sample_rate = 1000

t, h = rrc_filter(0, 300, 1/symbol_rate, sample_rate)
freqs = fftfreq(h.size, d=(1/sample_rate))
fh = fft(h)

plt.subplot(2,1,1)
plt.plot(t,h)
plt.title('Impulse Response of the RRC Filter in the Time Domain')

plt.subplot(2,1,2)
plt.plot(freqs,fh)
plt.title('Impulse Response of the RRC Filter in the Frequency Domain')

plt.show()