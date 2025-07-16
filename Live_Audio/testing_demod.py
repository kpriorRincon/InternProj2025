import numpy as np
import SigGen as SigGen
from decode_16qam import *
from scipy.signal import fftconvolve
import scipy.io.wavfile
import matplotlib.pyplot as plt
import wave as wv

CHUNK = 1024
sample_rate = 44100
sig_gen = SigGen.SigGen(910e6, 1)

reconstructed_bytes = bytearray()

with wv.open('output.wav', 'rb') as wave_file:
    while True:
        d = wave_file.readframes(CHUNK)
        if not d:
            break

        # Convert to bits
        d_bits = np.unpackbits(np.frombuffer(d, dtype=np.uint8))

        # Modulate
        t, qam_sig = sig_gen.generate_16QAM(d_bits)

        # Mix to baseband
        baseband_signal = qam_sig * np.exp(-1j * 910e6 * 2 * np.pi * t)
        rrc_added = fftconvolve(baseband_signal, sig_gen.h, mode='same')

        # Demodulate
        bits_out = signal_to_bits(rrc_added)

        # Pack bits into bytes
        byte_array_out = np.packbits(bits_out)
        reconstructed_bytes.extend(byte_array_out.tobytes())

# Convert accumulated bytes to int16 samples
samples = np.frombuffer(reconstructed_bytes, dtype='<i2')

# Write to output WAV file
scipy.io.wavfile.write('reconstructed.wav', sample_rate, samples)