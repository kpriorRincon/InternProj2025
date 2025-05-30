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

        # Properly cast input values to the correct types
        freq_val = int(freq.value)
        amp_val = int(amp.value)
        symbol_rate_val = int(symbol_rate.value)
        #create sig_gen object with the parameters from gui
        sig_gen = SigGen.SigGen(freq=freq_val, amp=amp_val, sample_rate = 20*freq_val, symbol_rate = symbol_rate_val)
        # Get the message from the input field and convert it to a bit sequence
        message = message_input.value
        # Convert the message to a binary bit sequence using the SigGen class
        bit_sequence = sig_gen.message_to_bits(message)
        if not all(bit in (0, 1) for bit in bit_sequence) and len(bit_sequence) % 2 != 0:
            # Check if the bit sequence is valid (only contains 0s and 1s and has an even length)
            ui.notify('Please enter a valid even number of bits (0s and 1s).')
            return
        # Generate QPSK waveform using the SigGen class
        t, qpsk_waveform, t_vertical_lines, symbols = sig_gen.generate_qpsk(bit_sequence)
        # Plot the waveform
        with ui.matplotlib(figsize=(20, 4)) as fig:
        
            plt.plot(t, qpsk_waveform)
            plt.ylim(-1.5/np.sqrt(2)*sig_gen.amp, 1.5/np.sqrt(2)*sig_gen.amp)
            for lines in t_vertical_lines:
                #add vertical lines at the symbol boundaries
                if lines < len(t):
                    plt.axvline(x=lines, color='black', linestyle='--', linewidth=1)

                    #add annotation for the symbol e.g. '00', '01', '10', '11'
                    # Reverse mapping: symbol -> binary pair
                    symbol = symbols[t_vertical_lines.index(lines)]
                    # Reverse the mapping to get binary pair from symbol
                    reverse_mapping = {v: k for k, v in sig_gen.mapping.items()}
                    binary_pair = reverse_mapping.get(symbol, '')
                    formatted_pair =str(binary_pair).replace("(", "").replace(")", "").replace(", ", "")
                    #debug
                    #print(formatted_pair)
                    x_dist = 1 / (2.7 * sig_gen.symbol_rate) #half the symbol period 
                    y_dist = 0.807*sig_gen.amp # 0.807 is the amplitude of the QPSK waveform
                    plt.annotate(formatted_pair, xy=(lines, 0), xytext=(lines + x_dist, y_dist), fontsize=17)
            plt.title(f'QPSK Waveform for {message}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()
            plt.show()
    #run the UI
    ui.run(native=True)

main()

