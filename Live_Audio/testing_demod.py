import numpy as np
import SigGen as SigGen
from decode_16qam import *
from scipy.signal import fftconvolve
import scipy.io.wavfile
import matplotlib.pyplot as plt
import wave as wv
CHUNK = 1024**2
with wv.open('output.wav', 'rb') as wave_file:
    d = wave_file.readframes(CHUNK)

d_bits = np.unpackbits(np.frombuffer(d, dtype = np.uint8))
print(d_bits)
#limiting the amplitude so we can test quantization
sig_gen = SigGen.SigGen(910e6, 1)
#get the bits from wav file
t, qam_sig = sig_gen.generate_16QAM(d_bits)


#tune that shit to baseband
baseband_signal = qam_sig * np.exp(-1j * 910e6 * 2 * np.pi * t)
rrc_added = fftconvolve(baseband_signal, sig_gen.h, mode = 'same')

bits_out = signal_to_bits(rrc_added)
print(f'bits out {bits_out}')

#reconstruct the bits into a wave file
byte_array_out = np.packbits(bits_out)
raw_bytes = byte_array_out.tobytes()
samples = np.frombuffer(raw_bytes, dtype='<i2')
sample_rate = 44100  # or use the original sample rate
scipy.io.wavfile.write('reconstructed.wav', sample_rate, samples)