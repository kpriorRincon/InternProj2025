import numpy as np
from scipy.signal import fftconvolve
import modulator as md
from commpy import rcosfilter

# Define start and end sequences
# 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 0 0 0 1 0 1 0 1 1 1 0 1 1 0 0 0 1
start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                1, 0, 1, 0, 0, 1, 0, 0,
                0, 0, 1, 0, 1, 0, 1, 1,
                1, 0, 1, 1, 0, 0, 0, 1]

# 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0 1 0 0 1 0
end_sequence = [0, 0, 1, 0, 0, 1, 1, 0,
                1, 0, 0, 0, 0, 0, 1, 0, 
                0, 0, 1, 1, 1, 1, 0, 1, 
                0, 0, 0, 1, 0, 0, 1, 0]

start_symbols = md.generate_qpsk(start_sequence)
end_symbols = md.generate_qpsk(end_sequence)
        
def correlator(rx_signal, sps):
    """
    Correlates the received signal with a known
    start and end sequence to detect the desired signal
    
    Parameters:
        rx_signal:          received complex signal array
        known_signal:       known complex signal array
        correlated signal:  complex array resulting
                            from the correlation
    
    Author: Trevor Wiseman
    """

    # Upsample
    start_upsampled = np.zeros(len(start_symbols)*sps, dtype=np.complex64)
    start_upsampled[::sps] = start_symbols
    end_upsampled = np.zeros(len(end_symbols)*sps, dtype=np.complex64)
    end_upsampled[::sps] = end_symbols 

    # RRC filter
    taps = 31
    beta = 0.35
    samp_rate = 32000
    _, h = rcosfilter(taps, beta, sps / samp_rate, samp_rate)

    # Pulse shape
    start_shaped = fftconvolve(start_upsampled, h, mode='full')
    end_shaped = fftconvolve(end_upsampled, h, mode='full')

    start_index = 0
    end_index = len(rx_signal) - 1

    # Correlate with start sequence
    correlated_signal = fftconvolve(rx_signal, np.conj(np.flip(start_shaped)), mode='full')
    end_cor_signal = fftconvolve(rx_signal, np.conj(np.flip(end_shaped)), mode='full')
    
    # Find maximum correlation
    start_index = np.argmax(np.abs(correlated_signal)) - 16*sps*len(h) # go back 16 symbols e.g. 32 bits
    end_index = np.argmax(np.abs(end_cor_signal))
    
    # error check
    if start_index >= end_index:
        start_index = 0
        end_index = len(rx_signal) - 1
        print("Start index is greater than end index...\nUsing default values")
    
    # index check
    print("Start index: ", start_index)
    print("End index: ", end_index)

    return start_index, end_index
