import numpy as np
from scipy.signal import fftconvolve
import modulator as md

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

    start_index = 0
    end_index = len(rx_signal) - 1
    
    # normalize rx_signal
    rx_signal = rx_signal / np.linalg.norm(rx_signal)

    # Correlate with start sequence
    correlated_signal = fftconvolve(rx_signal, np.conj(np.flip(start_symbols)), mode='full')
    end_cor_signal = fftconvolve(rx_signal, np.conj(np.flip(end_symbols)), mode='full')
    
    # Find maximum correlation
    start_index = np.argmax(np.abs(correlated_signal)) - 16*sps # go back 16 symbols e.g. 32 bits
    end_index = np.argmax(np.abs(end_cor_signal))

    if start_index > end_index:
        print("Signal not found! \nUsing default values...")
        start_index = 0
        end_index = len(rx_signal) - 1
    
    return start_index, end_index