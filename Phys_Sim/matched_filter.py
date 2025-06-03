import numpy as np
import sim_qpsk_noisy_demod as sim_qpsk

def rectangular_pulse(samples_per_symbol):
    return np.ones(samples_per_symbol)

def matched_filter(pulse_shape):
    return np.conj(pulse_shape[::-1])

from scipy.signal import lfilter

def apply_matched_filter(received_signal, pulse_shape):
    mf = matched_filter(pulse_shape)
    filtered = lfilter(mf, 1.0, received_signal)
    return filtered

samples_per_symbol = 10
pulse = rectangular_pulse(samples_per_symbol)

# Simulate transmit signal
tx_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=100)
tx_signal = np.zeros(len(tx_symbols) * samples_per_symbol, dtype=complex)
tx_signal[::samples_per_symbol] = tx_symbols
tx_waveform = np.convolve(tx_signal, pulse)

# Matched filtering
rx_waveform = apply_matched_filter(tx_waveform, pulse)

# Symbol sampling (ideal: sample at filter delay)
delay = len(pulse) - 1
rx_samples = rx_waveform[delay::samples_per_symbol]

# print results
tx_bits = sim_qpsk.bit_reader(tx_symbols)
rx_bits = sim_qpsk.bit_reader(rx_samples)
print("Transmitted Symbols:", tx_bits)
print("Received Samples:", rx_bits)