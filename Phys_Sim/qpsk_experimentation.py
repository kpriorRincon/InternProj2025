"""
Author: Kobe Prior
Date: May 28, 2025
Purpose: Experiment with generating QPSK waveform with a sequence of 4 bits
"""
import numpy as np
import matplotlib.pyplot as plt

def generate_qpsk_waveform(bits, frequency, sample_rate):
    """
    Generate a QPSK waveform from a sequence of bits.
    Args:
        bit_sequence (list): A list of bits (0s and 1s) to modulate.  
    Returns:    
        tuple: Time vector and QPSK waveform. 
    """
    # TODO - Implement QPSK modulation
    if len(bits) % 2 != 0:
        raise ValueError("Bit sequence must have an even length for QPSK modulation.")
    # Map bits to QPSK symbols
    symbols = []
    # odd bits are I (In-phase), even bits are Q (Quadrature)
    for i in range(0, len(bits), 2):
        if bits[i] == 0 and bits[i + 1] == 0:
            symbols.append(1 + 0j)  # 00 -> (1, 0)
        elif bits[i] == 0 and bits[i + 1] == 1:
            symbols.append(0 + 1j)  # 01 -> (0, 1)
        elif bits[i] == 1 and bits[i + 1] == 0:
            symbols.append(-1 + 0j) # 10 -> (-1, 0)
        elif bits[i] == 1 and bits[i + 1] == 1:
            symbols.append(0 - 1j) # 11 -> (0, -1)
    # Convert symbols to numpy array
    symbols = np.array(symbols)
    # Define time vector
    duration = len(symbols) / frequency
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate QPSK waveform
    qpsk_waveform = np.zeros_like(t, dtype=np.complex_)
    symbol_duration = len(t) // len(symbols)

def plot(qpsk_output):
    """
    Plot the QPSK waveform.
    Args:
        qpsk_output (tuple): A tuple containing time vector and QPSK waveform.
    """
    t, waveform = qpsk_output
    plt.figure(figsize=(10, 4))
    plt.plot(t, waveform)
    plt.title('QPSK Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

def main():
    # Example bit sequence (must be even length)
    bit_sequence = [0, 0, 0, 1, 1, 0, 1, 1]  # Represents two QPSK symbols
    
    # Generate QPSK waveform
    t, qpsk_waveform = generate_qpsk_waveform()
    
    # Plot the waveform
    plot_waveform(t, qpsk_waveform)
if __name__ == "__main__":
    main()

