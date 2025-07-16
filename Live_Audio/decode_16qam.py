from config import *
import numpy as np

def decimate(signal):
    return signal[::SPS]

def normalize_16QAM(signal):
    avg_power = np.mean(np.abs(signal)**2)
    normalized_signal = signal/avg_power
    return normalized_signal

def demodulate(ready_sig):
    #here we need to create a decision rule 
    a = np.array([-3,-1,1,3])/np.sqrt(10)
    
    
    return bits
