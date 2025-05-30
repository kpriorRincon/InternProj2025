"""
Author: Kobe Prior
Date: May 28, 2025
Purpose: Experiment with generating QPSK waveform with a sequence of 4 bits
"""
import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui
import Sig_Gen as SigGen
#create a sig gen object
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

    user_input = ui.input(label='Enter a message: ',
                          placeholder = 'e.g., hello')
    sumbit_button = ui.button('submit', on_click=lambda: use_data())
    def use_data():
        message = user_input.value
        #convert message to binary using Skylar's code: 
        message_binary = ''.join(format(ord(x), '08b') for x in message)
        # Convert string input to list of integers
        bit_sequence = [int(bit) for bit in message_binary.strip()]
        if not all(bit in (0, 1) for bit in bit_sequence):
            ui.notify('Please enter a valid even number of bits (0s and 1s).')
            return
        #create sig gen object
        sig_gen = SigGen.SigGen(freq=1000, amp=1.0)
        # Generate QPSK waveform using the SigGen class
        t, qpsk_waveform = sig_gen.generate_qpsk(bit_sequence, sample_rate = 1000000, symbol_rate = 1000)
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

