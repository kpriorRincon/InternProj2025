# =============================================================================
# File: main.py
# Authors: Skylar Harris, Jorge Hernandez, Kobe Prior, Trevor Wiseman
# Description:
#   This program provides a graphical user interface (GUI) for simulating a 
#   digital communication system, including a signal generator, repeater, and 
#   receiver. Users can input messages, configure simulation parameters such as 
#   frequency, gain, and noise, and visualize the signal processing stages. 
#   The application is built using NiceGUI and integrates with custom signal 
#   processing modules (Sig_Gen, Receiver, Repeater). It is intended for 
#   educational and prototyping purposes in the context of physical layer 
#   communications.
# =============================================================================

#imports
from nicegui import ui
import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater
import numpy as np
import matplotlib.pyplot as plt
import pickle
import subprocess


#global objects for the Sig_Gen, Receiver, and Repeater classes
symbol_rate = 10e6
sample_rate = 4e9
sig_gen_mapping ={
            (0, 0): (1 + 1j) / np.sqrt(2),
            (0, 1): (-1 + 1j) / np.sqrt(2),
            (1, 1): (-1 - 1j) / np.sqrt(2),
            (1, 0): (1 - 1j) / np.sqrt(2)
        }

t = None
tx_signal = None
tx_vert_lines = None
tx_symbols = None
tx_message_binary = None
rep_incoming_signal = None
rep_mixed_signal = None
rep_filtered_signal= None
rep_outgoing_signal = None
rx_message_binary = None
rx_recovered_message = None
rx_incoming_signal = None
rx_filtered_signal = None
rx_analytical_signal = None
f_in = None
f_out = None

noise_bool = False  # Global variable to control noise addition
noise_power = 0.1  # Default noise power
message_input = None  # Global variable to store the message input field

rx_message_binary = None
decoded_string = None

def Noise_Addr(input_wave, noise_power):
    #define noise
    noise =np.random.normal(0,noise_power,len(input_wave))
    return input_wave+noise
#front page
with ui.row().style('height: 100vh; width: 100%; display: flex; justify-content: center; align-items: center;'):
    with ui.link(target='\simulate_page'):
        simulate_button = ui.image('media/simulate.png').style('width: 50vh; height: 50vh;')
    with ui.link(target='#'):
        control_button = ui.image('media/control.png').style('width: 50vh; height: 50vh;')
        control_button.on('click', lambda: ui.notify('Control feature not yet available!'))  # Placeholder for control functionality

#simulate page
@ui.page('/simulate_page')
def simulate_page():
    """This function creates the simulation page where the user can select a simulation type and input parameters."""
    simulation_container = ui.column().style('order: 2;')
    with ui.row().style('justify-content: center;'):
        ui.label('Simulation Type').style('font-size: 2em; font-weight: bold;')

    #function to handle drop down menu
    def open_simulation_single_message():
        """This function triggers when the user selects a simulation type from the dropdown.
        It clears the simulation container and displays the appropriate input fields based on the selected type."""
        simulation_container
        selected_type = simulation_type_dropdown.value
        simulation_container.clear()
        if selected_type == 'Single Message':
            with simulation_container:
                ui.label('Single Message Simulation').style('font-size: 2em; font-weight: bold; ')
                ui.label('Enter message to be sent')
                message = ui.input(placeholder="hello world", value="hello world")
                ui.label('Simulation Parameters').style('font-size: 2em; font-weight: bold;')
                # When the user selects a simulation type, the parameters will change accordingly
                ui.label('Frequency In (MHz)',).style('width: 200px; margin-bottom: 10px;')
                freq_in_slider = ui.slider(min=902, max=928, step=1, value=905).props('label-always')
                ui.label('Frequency Out (MHz)').style('width: 200px; margin-bottom: 10px;')
                freq_out_slider = ui.slider(min=902, max=928, step=1, value=910).props('label-always')
                ui.label('Gain (dB)').style('width: 200px;margin-bottom: 10px;')
                gain_slider = ui.slider(min=0, max=10, step=1, value=0).props('label-always')
                #check box to ask if the user wants to add noise
                ui.label('Add Noise?').style('width: 200px;')
                noise_checkbox = ui.checkbox('add noise')#the value here will be a bool that can be used for siGen
                #iff the user checks the noise checkbox, then show the noise slider
                ui.label('Noise Level (dB)').style('width: 200px; margin-bottom: 10px;').bind_visibility_from(noise_checkbox, 'value')
                noise_slider = ui.slider(min=-60, max=5, step=1).props('label-always').bind_visibility_from(noise_checkbox, 'value')

                #submit button
                ui.button("Submit", on_click=lambda: start_sim()).style('width: 200px; height: 10px;')
                #function to store data on submit button click
                
                def start_sim():
                    """
                    stoample_rate = sample_rate, symbol_rate = symbol_ratee_data()
                    This function is triggered when the user clicks the submit button.
                    It collects the values from the input fields and stores them for further processing.
                    """
                    
                    global message_input
                    global rx_message_binary
                    global decoded_string
                    global t
                    global tx_signal
                    global tx_vert_lines
                    global tx_symbols 
                    global tx_message_binary
                    global rep_incoming_signal 
                    global rep_mixed_signal 
                    global rep_filtered_signal
                    global rep_outgoing_signal 
                    global rx_message_binary 
                    global rx_recovered_message 
                    global rx_incoming_signal 
                    global rx_filtered_signal 
                    global rx_analytical_signal
                    global f_in
                    global f_out

                    message_input = message.value
                    print(f'message input type is :{type(message_input)}')
                    # implement zmq here

                    if noise_checkbox.value:
                        global noise_bool
                        noise_bool = True
                        global noise_power
                    else:
                        noise_bool = False
                        noise_power = 0  # Default value if no noise is added

                    data_dict = {
                        'fin':freq_in_slider.value*1e6,
                        'fout': freq_out_slider.value*1e6, 
                        'message': message_input,
                        'sample rate': sample_rate,
                        'gain': 10**(int(gain_slider.value)/10), 
                        'symbol rate': symbol_rate, 
                        'noise_bool': noise_bool,
                        'noise_power': noise_power,
                    }
                    print(data_dict)
                    with open('data_dict.pkl', 'wb') as outfile:
                            pickle.dump(data_dict,outfile)
                    #basically run everything in zmq_integration

                    scripts = [['python', 'zmq_transmitter.py'],
                               ['python', 'zmq_repeater.py'],
                               ['python', 'zmq_receiver.py'],
                               ['python', 'zmq_controller.py']]

                    processes = []

                    for script in scripts:
                        process = subprocess.Popen(script)
                        processes.append(process)

                    for process in processes:
                        process.wait()

                    for process in processes:
                        process.terminate()
                    with open('controller_data.pkl', 'rb') as infile:
                        data = pickle.load(infile)

                    t = data['time']
                    tx_signal = data['transmitter signal']
                    tx_vert_lines = data['transmitter vertical lines']
                    tx_symbols = data['transmitter symbols']
                    tx_message_binary = data['transmitter message in binary']
                    rep_incoming_signal = data['repeater incoming signal']
                    rep_mixed_signal = data['repeater mixed signal']
                    rep_filtered_signal = data['repeater filtered signal']
                    rep_outgoing_signal = data['repeater outgoing signal']
                    rx_message_binary = data['receiver message in binary']
                    rx_recovered_message = data['receiver recovered message']
                    rx_incoming_signal = data['receiver incoming signal']
                    rx_filtered_signal = data['receiver filtered signal']
                    rx_analytical_signal = data['receiver analytical signal']
                    f_in = data['freq in']
                    f_out = data['freq out']
                    #debug:
                    print(tx_signal)

                    #all data read do plots here
                    #generate_plots

                    #this plot is for time qpsk
                    plt.figure(figsize=(15, 5))
                    plt.plot(t, tx_signal)
                    plt.ylim(-1/np.sqrt(2)*1-.5, 1/np.sqrt(2)*1+.5)

                    #if there are more than 10 symbols only show the first ten symbols
                    if len(tx_vert_lines) > 10:
                        plt.xlim(0, 10/symbol_rate)  # Show first 10 symbol periods
                    #if not don't touch the xlim

                    for lines in tx_vert_lines:
                        #add vertical lines at the symbol boundaries
                        if len(tx_vert_lines) > 10:
                            if lines < 10/symbol_rate:
                                plt.axvline(x=lines, color='black', linestyle='--', linewidth=1)

                                #add annotation for the symbol e.g. '00', '01', '10', '11'
                                # Reverse mapping: symbol -> binary pair
                                symbol = tx_symbols[tx_vert_lines.index(lines)]
                                # Reverse the mapping to get binary pair from symbol
                                reverse_mapping = {v: k for k, v in sig_gen_mapping.items()}
                                binary_pair = reverse_mapping.get(symbol, '')
                                formatted_pair =str(binary_pair).replace("(", "").replace(")", "").replace(", ", "")
                                #debug
                                #print(formatted_pair)
                                x_dist = 1 / (2.7 * symbol_rate) #half the symbol period 
                                y_dist = 0.707*1 + .2 # 0.807 is the amplitude of the QPSK waveform
                                plt.annotate(formatted_pair, xy=(lines, 0), xytext=(lines + x_dist, y_dist), fontsize=17)  
                        else:
                            if lines < len(t):
                                plt.axvline(x=lines, color='black', linestyle='--', linewidth=1)

                                #add annotation for the symbol e.g. '00', '01', '10', '11'
                                # Reverse mapping: symbol -> binary pair
                                symbol = tx_symbols[tx_vert_lines.index(lines)]
                                # Reverse the mapping to get binary pair from symbol
                                reverse_mapping = {v: k for k, v in sig_gen_mapping.items()}
                                binary_pair = reverse_mapping.get(symbol, '')
                                formatted_pair =str(binary_pair).replace("(", "").replace(")", "").replace(", ", "")
                                #debug
                                #print(formatted_pair)
                                x_dist = 1 / (2.7 * symbol_rate) #half the symbol period 
                                y_dist = 0.707*1 + .2 # 0.807 is the amplitude of the QPSK waveform
                                plt.annotate(formatted_pair, xy=(lines, 0), xytext=(lines + x_dist, y_dist), fontsize=17)
                            
                    if len(tx_vert_lines) > 10:
                        plt.title(f'QPSK Waveform for \"{message}\" (first 10 symbol periods)')
                    else:
                        plt.title(f'QPSK Waveform for \"{message}\"')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Amplitude')
                    plt.grid()
                    # Save the plot to a file
                    plt.savefig(f'qpsk_sig_gen/1_qpsk_waveform.png', dpi=300)    
                    #print("Debug: plot generated")
                    #END of time plot

                    #frequency for qpsk tx
                    #same figure size as above
                    n = len(t)
                    freqs = np.fft.fftfreq(n, d=1/sample_rate)
                    # FFT of original and shifted signals
                    fft = np.fft.fft(tx_signal)
                    fft_db = 20 * np.log10(np.abs(fft))
                    # get fft of qpsk signal
                    
                    plt.figure(figsize=(15, 5))
                    plt.plot(freqs,fft_db)
                    plt.title("FFT of QPSK signal")
                    plt.xlim(0, 1000e6)
                    plt.ylim(0, np.max(fft_db)+10)
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude (dB)')
                    #save plot
                    plt.savefig(f'qpsk_sig_gen/2_qpsk_waveform.png', dpi = 300)

                    #end plot for qpsk

                    #start plots for repeater: 
                    x_t_lim = 3 / symbol_rate
                    n = len(t)
                    freqs = np.fft.fftfreq(n, d=1/sample_rate)
                    positive_freqs = freqs > 0
                    positive_freq_values = freqs[positive_freqs]

                    # FFT of original and shifted signals
                    fft_input = np.fft.fft(rep_incoming_signal)
                    fft_shifted = np.fft.fft(rep_mixed_signal)
                    fft_filtered = np.fft.fft(rep_filtered_signal)
                    # Convert magnitude to dB
                    mag_input = 20 * np.log10(np.abs(fft_input))
                    mag_shifted = 20 * np.log10(np.abs(fft_shifted))
                    mag_filtered = 20 * np.log10(np.abs(fft_filtered))
                    plt.figure(figsize=(20, 6))

                    # --- Time-domain plot: Original QPSK ---
                    plt.subplot(1, 2, 1)
                    plt.plot(t, np.real(rep_incoming_signal))  # convert time to microseconds
                    plt.title("Original QPSK Signal (Time Domain)")
                    plt.xlabel("Time (μs)")
                    plt.ylabel("Amplitude")
                    plt.xlim(0, x_t_lim)
                    plt.grid(True)

                    plt.subplot(1, 2, 2)
                    positive_mags = mag_input[positive_freqs]
                    positive_freq_values = freqs[positive_freqs]
                    peak_index = np.argmax(positive_mags)
                    peak_freq = positive_freq_values[peak_index]
                    peak_mag = positive_mags[peak_index]
                    plt.plot(freqs, mag_input, label="Original QPSK", alpha=0.8)
                    plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq/1e6:.1f} MHz')
                    plt.text(peak_freq, peak_mag + 5, f'{peak_freq/1e6:.1f} MHz', color='r', ha='center')
                    #plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
                    plt.xlabel("Frequency (GHz)")
                    plt.ylabel("Magnitude (dB)")
                    plt.title("FFT of QPSK Before Frequency Shift")
                    plt.xlim(0, sample_rate / 2)  # From 0 to fs in MHz
                    plt.ylim(0, np.max(mag_input) + 10)
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('original_qpsk_rp.png')

                    plt.clf()

                    # --- Time-domain plot: Shifted QPSK ---
                    plt.subplot(1, 2, 1)
                    plt.plot(t, np.real(rep_mixed_signal))
                    plt.title("Shifted QPSK Signal (Time Domain)")
                    plt.xlabel("Time (μs)")
                    plt.ylabel("Amplitude")
                    plt.xlim(0, x_t_lim)
                    plt.grid(True)

                    

                    plt.subplot(1, 2, 2)
                    positive_mags = mag_shifted[positive_freqs]
                    positive_freq_values = freqs[positive_freqs]
                    peak_index = np.argmax(positive_mags)
                    peak_freq = positive_freq_values[peak_index]
                    peak_mag = positive_mags[peak_index]
                    plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
                    plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq/1e6:.1f} MHz')
                    plt.text(peak_freq, peak_mag + 5, f'{peak_freq/1e6:.1f} MHz', color='r', ha='center')
                    plt.xlabel("Frequency (GHz)")
                    plt.ylabel("Magnitude (dB)")
                    plt.title("FFT of QPSK After Frequency Shift")
                    plt.xlim(0, sample_rate / 2)  # From 0 to fs in MHz
                    plt.ylim(0, np.max(mag_input) + 10)
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()

                    plt.savefig('shifted_qpsk_rp.png')
                    plt.clf()

                    plt.subplot(1, 2, 1)
                    plt.plot(t, np.real(rep_filtered_signal))
                    plt.title("Filtered QPSK Signal (Time Domain)")
                    plt.xlabel("Time (μs)")
                    plt.ylabel("Amplitude")
                    plt.xlim(0, x_t_lim)
                    plt.grid(True)

                    plt.subplot(1, 2, 2)
                    positive_mags = mag_filtered[positive_freqs]
                    positive_freq_values = freqs[positive_freqs]
                    peak_index = np.argmax(positive_mags)
                    peak_freq = positive_freq_values[peak_index]
                    peak_mag = positive_mags[peak_index]
                    #print(freqs[peak_index-3:peak_index+3])
                    plt.plot(freqs, mag_filtered, label="Filtered QPSK", alpha=0.8)
                    plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq/1e6:.1f} MHz')
                    plt.text(peak_freq, peak_mag + 5, f'{peak_freq/1e6:.1f} MHz', color='r', ha='center')
                    plt.xlabel("Frequency (GHz)")
                    plt.ylabel("Magnitude (dB)")
                    plt.title("FFT of QPSK After Filtering")
                    plt.xlim(0, sample_rate / 2)  # From 0 to fs in MHz
                    plt.ylim(0, np.max(mag_input) + 10)
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()

                    plt.savefig('filtered_qpsk_rp.png')
                    plt.clf()
                    #end repeater plotting


                    #start receiver plotting:
                    # constellation plot
                    plt.figure(figsize=(10, 4))
                    plt.plot(np.real(rx_analytical_signal), np.imag(rx_analytical_signal), '.')
                    plt.grid(True)
                    plt.title('Constellation Plot of Sampled Symbols')
                    plt.xlabel('Real')
                    plt.ylabel('Imaginary')
                    plt.savefig('demod_media/Constellation.png')

                    # Plot the waveform and phase
                    plt.figure(figsize=(10, 4))
                    plt.plot(t, rx_analytical_signal.real, label='I (real part)')
                    plt.plot(t, rx_analytical_signal.imag, label='Q (imag part)')
                    plt.title('Hilbert Transformed Waveform (Real and Imag Parts)')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Amplitude')
                    plt.grid()
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('demod_media/Base_Band_Waveform.png')

                    # plot the fft
                    ao_fft = np.fft.fft(rx_analytical_signal)
                    freqs = np.fft.fftfreq(len(rx_analytical_signal), d=1/2*sample_rate)
                    plt.figure(figsize=(10, 4))
                    plt.plot(freqs, 20*np.log10(ao_fft))
                    plt.title('FFT of the Base Band Signal')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Madgnitude (dB)')
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig('demod_media/Base_Band_FFT.png')



                    
                    ui.notify('Data stored successfully!') 

            
        elif selected_type == 'Continuous Message': 
            with simulation_container:
                ui.notify("Continuous Message Simulation is not yet implemented.")

    choices = [
        'Single Message', 
        'Continuous Message'
    ]
    simulation_type_dropdown = ui.select(choices, on_change=open_simulation_single_message).style('width: 200px; height: 40px;')

    with ui.column().style('position: absolute; top: 500px; left: 700px; '):
        with ui.link(target='/signal_generator_page'):
            ui.image('media/antenna_graphic.png').style('width:200px;')
        ui.label("Signal Generator").style('font-size: 1.5em; font-weight: bold;')
    with ui.column().style('position: absolute; top: 20px; left: 1000px;'):
        ui.label("Repeater").style('font-size: 1.5em; font-weight: bold; margin-left: 55px;')
        with ui.link(target='/repeater_page'):
            ui.image('media/sattelite.png').style('width:300px;')

    with ui.column().style('position: absolute; top: 500px; left: 1500px;'): 
        with ui.link(target='/receiver_page'):
            ui.image('media/antenna_graphic_flipped.png').style('width:200px;')
        ui.label("Receiver").style('font-size: 1.5em; font-weight: bold; margin-left: 110px;')


#simulation Signal Generator page
@ui.page('/signal_generator_page')
def signal_generator_page():
    """This function creates the Signal Generator page where the user can view outputs from the signal generator."""
    ui.button('back', on_click=ui.navigate.back)
    #create the qpsk wave form from the message
    if message_input is not None:
        with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
            ui.label(f'Message entered: {message_input}').style('font-size: 2em; font-weight: bold;')
            #we want to show the header in a different color as the actual message 
            bit_sequence = tx_message_binary
            marker = ''
            payload = ''
            for i in range(len(bit_sequence)):
                #get the first 8 bits as the marker
                if i < 8:
                    marker += str(bit_sequence[i])
                else:
                    payload += str(bit_sequence[i])
            ui.label('Bit Sequence:').style('font-size: 1.5em; font-weight: bold;')
            ui.html(f'''<div style ="font-size: 1.5em; font-weight: bold; color: #D2042D;"><span style = 'color:#0072BD'>Marker</span> | <span style = 'color:black'>Message</span></div>''').style('text-align: center;')
            ui.html(f'''<div style ="font-size: 1.5em; font-weight: bold; color: #D2042D; text-wrap:wrap; word-break: break-all;"><span style = 'color:#0072BD'>{marker}</span> | <span style = 'color:black; '>{payload}</span></div>''').style('text-align: center;')

            #need to insure we get the most up to date image that's why we use .force_reload()
#simulation Repeater page
            ui.image('qpsk_sig_gen/1_qpsk_waveform.png').style('width: 70%; height: auto;').force_reload()
            ui.image('qpsk_sig_gen/2_qpsk_waveform.png').style('width: 70%; height: auto;').force_reload()
@ui.page('/repeater_page')
def repeater_page():

    """This function creates the repeater page where the user can view outputs from the repeater."""
    ui.button('back', on_click=ui.navigate.back)
    if message_input is not None:
        with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
            ui.label(f'Input Frequency: {f_in/1e6:.1f} MHz      Output Frequency: {f_out/1e6:.1f} MHz').style('font-size: 2em; font-weight: bold;')

    ui.image('original_qpsk_rp.png').force_reload()
    ui.image('shifted_qpsk_rp.png').force_reload()
    ui.image('filtered_qpsk_rp.png').force_reload()


#simulation receiver page
@ui.page('/receiver_page')
def receiver_page():
    """This function creates the Receiver page where the user can view outputs from the receiver."""
    ui.button('back', on_click=ui.navigate.back)
    #on this page put plots


    marker = ''
    payload = ''
    for i in range(len(rx_message_binary)):
        #get the first 8 bits as the marker
        if i < 8:
            marker += str(rx_message_binary[i])
        else:
            payload += str(rx_message_binary[i])
    with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
        ui.image('demod_media/Constellation.png').force_reload()
        ui.image('demod_media/Base_Band_Waveform.png').force_reload()
        ui.image('demod_media/Base_Band_FFT.png').force_reload()
        ui.label('Bit Sequence:').style('font-size: 1.5em; font-weight: bold;')
        ui.html(f'''<div style ="font-size: 1.5em; font-weight: bold; color: #D2042D;"><span style = 'color:#0072BD'>Marker</span> | <span style = 'color:black'>Message</span></div>''').style('text-align: center;')
        ui.html(f'''<div style ="font-size: 1.5em; font-weight: bold; color: #D2042D; text-wrap:wrap; word-break: break-all;"><span style = 'color:#0072BD'>{marker}</span> | <span style = 'color:black; '>{payload}</span></div>''').style('text-align: center;')
        ui.label(f'Decoded Message: {rx_recovered_message}')

    pass



# TODO implement the control page when we are able to
#control page
@ui.page('/control_page')
def control_page():
    pass


ui.run()    
