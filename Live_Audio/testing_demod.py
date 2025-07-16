import numpy as np
import SigGen as SigGen
from decode_16qam import *
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import wave

#limiting the amplitude so we can test quantization
sig_gen = SigGen.SigGen(910e6, 1)
#get the bits from wav file
with wave.open('output.wav')
bits = [1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1]
t, qam_sig = sig_gen.generate_16QAM(bits)


#tune that shit to baseband
baseband_signal = qam_sig * np.exp(-1j * 910e6 * 2 * np.pi * t)
rrc_added = fftconvolve(baseband_signal, sig_gen.h, mode = 'same')

bits_out = signal_to_bits(rrc_added)
print(f'bits out {bits_out}')