from config import *
import numpy as np
import matplotlib.pyplot as plt
def decimate(signal):
    return signal[::SPS]

def normalize_16QAM(signal):
    #plots are for debug
    # plt.plot(np.real(signal), np.imag(signal), 'o')
    #find the average energy 
    avg_magnitude = np.mean(np.abs(signal))
    normalized_signal = signal / avg_magnitude
    # plt.plot(np.real(normalized_signal), np.imag(normalized_signal), 'o')
    # plt.show()
    return normalized_signal

def decide(ready_sig):
    #here we need to create a decision rules
    #normalized levels
    a = np.array([-3,-1,1,3])/np.sqrt(10)

    def quantize(value):
        return a[np.argmin(np.abs(value - a))]
    
    #decisions = [(1+1j)/root(10), ...., (-1 -3j)/root(10)] essentially snaps symbols to the closest constellation
    decisions = []
    for symbol in ready_sig:
        real_part = quantize(np.real(symbol))
        imag_part = quantize(np.imag(symbol))
        complex_number = real_part + imag_part * 1j
        decisions.append(complex_number)
    # print(decisions)
    return decisions

def demodulate(decisions):
    #we need to get our decisions into bits
    INV_Map = {v: k for k, v in MAPPING.items()}
    bit_list = [INV_Map[sym] for sym in decisions]
    bits = []
    for sub_tuple in bit_list:
        bits.extend(sub_tuple)
    return bits

def signal_to_bits(signal):
    decimated = decimate(signal)
    #normalization doesn't seem to work
    normalized_sig = normalize_16QAM(decimated)
    decided_symbols = decide(normalized_sig)
    return demodulate(decided_symbols)
