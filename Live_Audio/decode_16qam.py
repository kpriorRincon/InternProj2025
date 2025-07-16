from config import *
import numpy as np

def decimate(signal):
    return signal[::SPS]

def normalize_16QAM(signal):
    avg_power = np.mean(np.abs(signal)**2)
    normalized_signal = signal/avg_power
    return normalized_signal
def decide(ready_sig):

    #here we need to create a decision rules
    #normalized levels
    a = np.array([-3,-1,1,3])/np.sqrt(10)
    def quantize(value):
        return a[np.argmin(np.abs(value - a))]
    #decisions = [(1+1j)/root(10), ...., (-1 -3j)/root(10)] essentially snaps symbols to the closest constellation
    decisions = [(quantize(np.real(symbol)) + 1j * quantize(np.imag(symbol)) for symbol in ready_sig)]
    return decisions

def demodulate(decisions):
    #we need to get our decisions into bits
    bits = 
    
    return bits
