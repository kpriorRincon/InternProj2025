"""
Author: Kobe Prior
Date: May 28, 2025
Purpose: Experiment with generating QPSK waveform with a sequence of 4 bits
"""
import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui

def generate_qpsk_waveform(bits, symbol_rate, sample_rate):
    """
    Generate a QPSK waveform from a sequence of bits.
    Args:
        bit_sequence (list of integers): A list of bits (0s and 1s) to modulate.  
    Returns:    
        tuple: Time vector and QPSK waveform. 
    """
    # TODO - Implement QPSK modulation
    carrier_frequency = symbol_rate * 2  # Example carrier frequency, can be adjusted
    if len(bits) % 2 != 0:
        raise ValueError("Bit sequence must have an even length.")

    # Map bit pairs to complex symbols
    mapping = {
        (0, 0): (1 + 1j) / np.sqrt(2),
        (0, 1): (-1 + 1j) / np.sqrt(2),
        (1, 1): (-1 - 1j) / np.sqrt(2),
        (1, 0): (1 - 1j) / np.sqrt(2)
    }
    #seperate into odd and even and map to complex symbols
    symbols = [mapping[(bits[i], bits[i + 1])] for i in range(0, len(bits), 2)]

    samples_per_symbol = int(sample_rate / symbol_rate)
    #time vector for the wave form defined start at 0 end at the length of the symbols times samples per symbol
    # and spaced by the sample rate
    t = np.arange(0, len(symbols) * samples_per_symbol) / sample_rate

    # Initialize the QPSK waveform
    qpsk_waveform = np.zeros_like(t)
    #for loop that tracks the index of the symbol and the symbol itself
    for i, symbol in enumerate(symbols):
        #compute the phase offset for the symbol
        phase_offset = np.angle(symbol)
        idx_start = i * samples_per_symbol
        idx_end = idx_start + samples_per_symbol
        time_slice = t[idx_start:idx_end] # time slice for the current symbol
        qpsk_waveform[idx_start:idx_end] = (
            #eqn from Haupt pg 86
            np.sqrt(2)*np.cos(2*carrier_frequency*np.pi * time_slice + phase_offset)
        )
    return t, qpsk_waveform


def main():
    """
    main()
    Launches a user interface for QPSK waveform generation and visualization.
    This function creates a simple GUI that prompts the user to enter a sequence of bits.
    Upon submission, it converts the input string into a list of integers, generates the
    corresponding QPSK waveform using the provided bit sequence, and displays the waveform
    using a matplotlib plot embedded in the UI.
    Workflow:
        1. Prompts the user for a bit sequence (e.g., '1100').
        2. Converts the input string to a list of integers.
        3. Generates the QPSK waveform using the specified symbol frequency and sample rate.
        4. Plots the generated waveform in the UI.
    Note:
        - Requires the `generate_qpsk_waveform` function to be defined elsewhere.
        - Assumes the presence of a UI framework with `ui.input`, `ui.button`, `ui.matplotlib`, and `ui.run`.
    """

    user_input = ui.input(label='Enter bits: ',
                          placeholder = 'e.g., 1100')
    user_input.props('type', 'text')
    sumbit_button = ui.button('submit', on_click=lambda: use_data())
    def use_data():
        bit_sequence = user_input.value
        # Convert string input to list of integers
        bit_sequence = [int(bit) for bit in bit_sequence.strip()]
        # Generate QPSK waveform
        t, qpsk_waveform = generate_qpsk_waveform(bit_sequence, symbol_frequency=1000, sample_rate=1000000)
        # Plot the waveform
        with ui.matplotlib(figsize=(20, 4)) as fig:
            plt.plot(t, qpsk_waveform, label='QPSK Waveform')
            plt.title('QPSK Waveform')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()
            plt.show()
    #run the UI
    ui.run()

main()

