"""
Author: Kobe Prior
Date: May 28, 2025
Purpose: 
    This program provides a graphical user interface (GUI) for generating and visualizing QPSK (Quadrature Phase Shift Keying) waveforms. 
    Users can input a text message, which is converted to a binary bit sequence and then modulated using QPSK. 
    The resulting waveform is displayed interactively using matplotlib within the GUI.
"""
import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Sig_Gen as SigGen

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

    message_input = ui.input(label='Enter a message: ',
                          placeholder = 'e.g., hello').style('width: 300px;')
    freq = ui.number(label='Frequency (Hz): ', placeholder=902000).style('width: 300px;')
    amp = ui.number(label='Amplitude: ', placeholder=1).style('width: 300px;')
    symbol_rate = ui.number(label='Symbol Rate (Hz): ', placeholder=1000).style('width: 300px;')
    # Create a button to submit the input
    # The button will trigger the use_data function when clicked
    ui.button('submit', on_click=lambda: use_data())
    def use_data():
        """
        use_data()
        This function is triggered when the user clicks the submit button.
        It retrieves the user input, converts it to a binary bit sequence using the SigGen class,
        generates the QPSK waveform, and plots it using matplotlib.
        Workflow:
            1. Retrieves the user input from the input field.
            2. Converts the input message to a binary bit sequence using the SigGen class.
            3. Validates the bit sequence to ensure it contains only 0s and 1s.
            4. Generates the QPSK waveform using the SigGen class.
            5. Plots the generated waveform with appropriate labels and grid.
        """

        #close any existing plots:
        plt.close('all')
        # Properly cast input values to the correct types
        freq_val = int(freq.value)
        amp_val = int(amp.value)
        symbol_rate_val = int(symbol_rate.value)
        #create sig_gen object with the parameters from gui
        sig_gen = SigGen.SigGen(10*freq_val, symbol_rate_val)
        sig_gen.amp = amp_val
        sig_gen.freq = freq_val
        # Get the message from the input field and convert it to a bit sequence
        message = message_input.value
        # Convert the message to a binary bit sequence using the SigGen class
        bit_sequence = sig_gen.message_to_bits(message)
        if not all(bit in (0, 1) for bit in bit_sequence) and len(bit_sequence) % 2 != 0:
            # Check if the bit sequence is valid (only contains 0s and 1s and has an even length)
            ui.notify('Please enter a valid even number of bits (0s and 1s).')
            return
        # Generate QPSK waveform using the SigGen class
        sig_gen.generate_qpsk(bit_sequence)
        # Plot the waveform
        sig_gen.plot_time_png(message)
        ui.image('1_qpsk_waveform.png').force_reload()
    #run the UI
    ui.run(native=True)

main()

