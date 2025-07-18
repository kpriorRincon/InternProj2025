from config import *
import numpy as np
import matplotlib.pyplot as plt

def decimate(signal):
    #extract only symbols from signal
    return signal[::SPS]

def normalize_16QAM(signal):
    #find the average magnitude
    avg_magnitude = np.mean(np.abs(signal))
    #this will scale a signal to ignore any attenuation
    normalized_signal = signal / avg_magnitude

    #plots are for debug
    # plt.plot(np.real(signal), np.imag(signal), 'o')
    # plt.plot(np.real(normalized_signal), np.imag(normalized_signal), 'o')
    # plt.show()
    return normalized_signal

def decide(ready_sig):
    #here we need to create a decision rules

    #normalized levels
    a = np.array([-3,-1,1,3])/np.sqrt(10)

    def quantize(value):
        #snaps any value to the closest level in I or Q
        return a[np.argmin(np.abs(value - a))]
    
    #decisions = [(1+1j)/root(10), ...., (-1 -3j)/root(10)] essentially snaps symbols to the closest constellation
    decisions = []
    for symbol in ready_sig:
        real_part = quantize(np.real(symbol))
        imag_part = quantize(np.imag(symbol))
        complex_number = real_part + imag_part * 1j
        decisions.append(complex_number)
    #debug
    # print(decisions)
    return decisions

def demodulate(decisions):
    #we need to get our decisions into bits
    INV_Map = {v: k for k, v in MAPPING.items()} #replaces the keys with values in mapping dictionary
    bit_list = [INV_Map[sym] for sym in decisions] #convert symbols to bits [(1,0,1,1), ..., (1,0,0,0)]
    bits = []
    for sub_tuple in bit_list:
        bits.extend(sub_tuple)
    #bits now in array [1, 0, 1, 1, ..., 1, 0, 0 , 0]
    return bits

def signal_to_bits(signal):
    #run the whole thing to demodulate qam
    decimated = decimate(signal)
    normalized_sig = normalize_16QAM(decimated)
    decided_symbols = decide(normalized_sig)
    return demodulate(decided_symbols)
