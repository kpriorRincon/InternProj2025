import numpy as np
import matplotlib.pyplot as plt
from Sig_Gen import rrc_filter
symbol_rate = 10
sample_rate = 40
#generate 10 random qpsk symbos
def gen_qpsk_symbols(n = 10):
    bits = np.random.randint(0,2,size = (n, 2))
    mapping={
        (0,0): 1+1j,
        (0,1): -1+1j,
        (1,1): -1-1j,
        (1,0): 1-1j
    }
    symbols = np.array([mapping[mapping[tuple(b)]] for b in bits]) / np.sqrt(2) #normalize energy

    def apply_pulseshaping (symbols):
        #upsample symbols to 4 sps
        samples_per_symbol = int(sample_rate/symbol_rate)

        upsampled_symbols = np.zeros(len(symbols)*samples_per_symbol, dtype = complex)
        upsampled_symbols[::samples_per_symbol] = symbols
        h,_ = rrc_filter(0.3, 101, 1/symbol_rate, sample_rate)
        shaped_signal = np.convolve(upsampled_symbols,h, mode='same')

        return shaped_signal
    
    if __name__ == "__main__":
        symbols = gen_qpsk_symbols(10)
        shaped_signal = apply_pulseshaping(symbols)
        plt.figure(figsize=(6, 4))
        plt.plot(np.real(shaped_signal))
        plt.show()