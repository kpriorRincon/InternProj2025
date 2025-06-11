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

# configure the RTL SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6 # Hz
sdr.center_freq = 920e6   # Hz
sdr.freq_correction = 60  # PPM
print(sdr.valid_gains_db)
sdr.gain = 49.6
print(sdr.gain)

# run the receiver code
total_samples = 2048
t = np.arange(total_samples) / sdr.sample_rate
x = sdr.read_samples(total_samples) # get rid of initial empty samples
symbol_rate = sdr.sample_rate / 5
signal, sampled_symbols, best_bits = ed.demodulator(x, sdr.sample_rate, symbol_rate, t, sdr.center_freq)

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