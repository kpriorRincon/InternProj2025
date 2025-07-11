"""
Author: Kobe Prior
Date: May 28, 2025
Purpose: Experiment with generating QPSK waveform with a sequence of 4 bits
"""
import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui
def generate_qpsk_waveform(bits, symbol_frequency, sample_rate):
    """
    Generate a QPSK waveform from a sequence of bits.
    Args:
        bit_sequence (list of integers): A list of bits (0s and 1s) to modulate.  
    Returns:    
        tuple: Time vector and QPSK waveform. 
    """
    # TODO - Implement QPSK modulation
    carrier_frequency = symbol_frequency * 2  # Example carrier frequency, can be adjusted
    if len(bits) % 2 != 0:
        raise ValueError("Bit sequence must have an even length.")

    # Map bit pairs to complex symbols
    mapping = {
        (0, 0): (1 + 1j) / np.sqrt(2),
        (0, 1): (-1 + 1j) / np.sqrt(2),
        (1, 1): (-1 - 1j) / np.sqrt(2),
        (1, 0): (1 - 1j) / np.sqrt(2)
    }
    symbols = [mapping[(bits[i], bits[i + 1])] for i in range(0, len(bits), 2)]

    samples_per_symbol = int(sample_rate / symbol_frequency)
    t = np.arange(0, len(symbols) * samples_per_symbol) / sample_rate
    qpsk_waveform = np.zeros_like(t)

    for i, symbol in enumerate(symbols):
        phase_offset = np.angle(symbol)  # Phase offset for the symbol
        idx_start = i * samples_per_symbol
        idx_end = idx_start + samples_per_symbol
        time_slice = t[idx_start:idx_end]
        qpsk_waveform[idx_start:idx_end] = (
            np.sqrt(2)*np.cos(2*carrier_frequency*np.pi * time_slice + phase_offset)
        )

    return t, qpsk_waveform

def plot(t, qpsk_output):
    """
    Plot the QPSK waveform.
    Args:
        qpsk_output (tuple): A tuple containing time vector and QPSK waveform.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(t, qpsk_output, label='QPSK Waveform')
    plt.title('QPSK Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

def demodulate_qpsk(waveform, frequency, sample_rate):
    """
    Demodulate a QPSK waveform to retrieve the original bit sequence.
    Args:
        waveform (np.ndarray): The QPSK waveform to demodulate.
        frequency (float): The frequency of the QPSK signal.
        sample_rate (int): The sample rate of the signal.
    Returns:
        list: The demodulated bit sequence.
    """
    # TODO - Implement QPSK demodulation
    pass

def main():
    # TODO - Implement GUI for user input
    bit_sequence = input("Enter a sequence of bits (e.g., 1100): ")
    
    print(type(bit_sequence))
    # Convert string input to list of integers
    bit_sequence = [int(bit) for bit in bit_sequence.strip()]    
    # Generate QPSK waveform
    t, qpsk_waveform = generate_qpsk_waveform(bit_sequence, symbol_frequency=1000, sample_rate=1000000)
    # Plot the waveform
    plot(t, qpsk_waveform)


if __name__ == "__main__":
    main()

