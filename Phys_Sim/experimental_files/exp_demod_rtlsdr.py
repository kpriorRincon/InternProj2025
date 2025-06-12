#################
#
# Author: Trevor
#
#################

# imports
from rtlsdr import RtlSdr
import exp_demodulator as ed
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import Sig_Gen_Noise as SigGen

# detection function
def detect(sig, sample_rate, symbol_rate):
    # define the detection threshold
    threshold = 0.7

    # Define start and end sequences
    start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                      1, 0, 1, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 1, 0, 1, 1,
                      1, 0, 1, 1, 0, 0, 0, 1]
    sig_gen = SigGen.SigGen(0, 1.0, sample_rate, symbol_rate)
    _, start, _, _ = sig_gen.generate_qpsk(start_sequence, False, 0.1)

    # detection algorithm
    sig = (sig - np.max(sig)) / (np.max(sig) - np.min(sig))
    cor_sig = fftconvolve(sig, np.conj(np.flip(start)))
    cor_sig = (cor_sig - np.max(cor_sig)) / (np.max(cor_sig) - np.min(cor_sig))
    cor_power = np.max(np.abs(cor_sig))
    if cor_power > threshold:
        start_cor = True
    
    return start_cor

# configure the RTL SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6 # Hz
sdr.center_freq = 920e6   # Hz
sdr.freq_correction = 60  # PPM
print(sdr.valid_gains_db)
sdr.gain = 49.6
print(sdr.gain)
total_samples = 2048
t = np.arange(total_samples) / sdr.sample_rate
symbol_rate = sdr.sample_rate / 5
start_cor = False
x = None

# run the detection algorithm continuously until signal is found
print("Searching for the signal...")
while start_cor == False:
    x = sdr.read_samples(total_samples) # get rid of initial empty samples
    start_cor = detect(x)
print("Signal found \ndemodulating...")

# after the signal is found demodulate and display
signal, sampled_symbols, best_bits = ed.demodulator(x, sdr.sample_rate, symbol_rate, t, sdr.center_freq)
print("bits: ", best_bits)
print("Message: ", ed.get_string(best_bits))

# plot the data
# constellation plot
fig, axs = plt.subplots(3,1)
axs[0].scatter(np.real(sampled_symbols), np.imag(sampled_symbols))
axs[0].grid(True)
axs[0].set_title('Constellation Plot of Sampled Symbols')
axs[0].set_xlabel('Real')
axs[0].set_ylabel('Imaginary')

# Plot the waveform and phase
axs[1].plot(np.arange(0,len(signal)), np.real(signal), label='I (real part)')
axs[1].plot(np.arange(0, len(signal)), np.imag(signal), label='Q (imag part)')
axs[1].set_title('Filtered Baseband Signal Time Signal (Real and Imag Parts)')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Amplitude')
axs[1].grid()
axs[1].legend()

# plot the fft
ao_fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), d=1/2*signal)
axs[2].plot(freqs, 20*np.log10(ao_fft))
axs[2].set_title('FFT of the Filtered Base Band Signal')
axs[2].set_xlabel('Frequency (Hz)')
axs[2].set_ylabel('Madgnitude (dB)')
axs[2].grid()

plt.show()