# =============================================================================
# File: Gui_zmq_controlled.py
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
import numpy as np
import matplotlib.pyplot as plt
import pickle
import subprocess
from Plotter import Plotter


#global objects for the Sig_Gen, Receiver, and Repeater classes
symbol_rate = 5e6
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
    with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
        ui.html('<u>Project Red Mountain</u>').style('font-size: 3em; font-weight: bold;')
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
        if selected_type == 'Basic Simulation':
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
                    
                    # implement zmq here

                    if noise_checkbox.value:
                        global noise_bool
                        noise_bool = True
                        global noise_power
                        noise_power = 10**(noise_slider.value/10)
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
                    tx_upsampled_symbols = data['transmitter upsampled symbols']
                    tx_message_binary = data['transmitter message in binary']
                    rep_incoming_signal = data['repeater incoming signal']
                    rep_mixed_signal = data['repeater mixed signal']
                    rep_outgoing_signal = data['repeater outgoing signal']
                    rx_message_binary = data['receiver message in binary']
                    rx_recovered_message = data['receiver recovered message']
                    rx_incoming_signal = data['receiver incoming signal']
                    rx_filtered_signal = data['receiver filtered signal']
                    rx_analytical_signal = data['receiver analytical signal']
                    f_in = data['freq in']
                    f_out = data['freq out']
                    rx_sampled_symbols = data['sampled symbols']

                    # print(rx_sampled_symbols)
                    
                    #all data read do plots here
                    #generate_plots

                    Plotter(sample_rate, t, tx_signal, tx_vert_lines, symbol_rate, tx_symbols, tx_upsampled_symbols,sig_gen_mapping, message_input, rep_incoming_signal, rep_mixed_signal, rx_incoming_signal, rx_filtered_signal, rx_analytical_signal, rx_sampled_symbols)
                    ui.notify('Data stored successfully!') 

            
        elif selected_type == 'Advanced Simulation': 
            with simulation_container:
                ui.notify("Advanced Simulation not yet ported")

    choices = [
        'Basic Simulation', 
        'Advanced Simulation'
    ]
    simulation_type_dropdown = ui.select(choices, on_change=open_simulation_single_message).style('width: 200px; height: 40px;')

    with ui.column().style('position: absolute; top: 600px; left: 700px; '):
        with ui.link(target='/signal_generator_page'):
            ui.image('media/antenna_graphic.png').style('width:200px;')
        ui.label("Signal Generator").style('font-size: 1.5em; font-weight: bold;')
    with ui.column().style('position: absolute; top: 120px; left: 1000px;'):
        ui.label("Repeater").style('font-size: 1.5em; font-weight: bold; margin-left: 55px;')
        with ui.link(target='/repeater_page'):
            ui.image('media/sattelite.png').style('width:300px;')

    with ui.column().style('position: absolute; top: 600px; left: 1500px;'): 
        with ui.link(target='/receiver_page'):
            ui.image('media/antenna_graphic_flipped.png').style('width:200px;')
        ui.label("Receiver").style('font-size: 1.5em; font-weight: bold; margin-left: 110px;')


#simulation Signal Generator page
@ui.page('/signal_generator_page')
def signal_generator_page():
    """This function creates the Signal Generator page where the user can view outputs from the signal generator."""
    ui.button('back', on_click=ui.navigate.back)
    with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
        ui.html('<u>Signal Generator Page</u>').style('font-size: 3em; font-weight: bold;')
    #create the qpsk wave form from the message
    if message_input is not None:
        with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
            ui.label(f'Message entered: {message_input}').style('font-size: 2em; font-weight: bold;')
            #we want to show the header in a different color as the actual message 
            bit_sequence = tx_message_binary
            marker_front = ''
            marker_back = ''
            payload = ''
            for i in range(len(bit_sequence)):
                #get the first 8 bits as the marker
                if i < 32:
                    marker_front += str(bit_sequence[i])
                elif i > len(bit_sequence) - 32:
                    marker_back += str(bit_sequence[i])
                else:
                    payload += str(bit_sequence[i])
            ui.label('Bit Sequence:').style('font-size: 1.5em; font-weight: bold;')
            ui.html(f'''<div style ="font-size: 1.5em; font-weight: bold; color: #D2042D;"><span style = 'color:#0072BD'>Marker</span> | <span style = 'color:black'>Message</span> | <span style = 'color:#0072BD'>Marker</span></div>''').style('text-align: center;')
            ui.html(f'''<div style ="font-size: 1.5em; font-weight: bold; color: #D2042D; text-wrap:wrap; word-break: break-all;"><span style = 'color:#0072BD'>{marker_front}</span> | <span style = 'color:black; '>{payload}</span> | <span style = 'color:#0072BD'>{marker_back}</span></div>''').style('text-align: center;')

            #need to insure we get the most up to date image that's why we use .force_reload()
            ui.image('qpsk_sig_gen/baseband.png').style('width: 70%; height: auto;').force_reload()
            ui.image('qpsk_sig_gen/1_qpsk_waveform.png').style('width: 70%; height: auto;').force_reload()
            ui.image('qpsk_sig_gen/2_qpsk_waveform.png').style('width: 70%; height: auto;').force_reload()

#simulation Repeater page
@ui.page('/repeater_page')
def repeater_page():
    with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
        ui.html('<u>Repeater Page</u>').style('font-size: 3em; font-weight: bold;')
    """This function creates the repeater page where the user can view outputs from the repeater."""
    ui.button('back', on_click=ui.navigate.back)
    if message_input is not None:
        with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
            ui.label(f'Input Frequency: {f_in/1e6:.1f} MHz      Output Frequency: {f_out/1e6:.1f} MHz').style('font-size: 2em; font-weight: bold;')

    ui.image('repeater_plots/original_qpsk_rp.png').force_reload()
    ui.image('repeater_plots/shifted_qpsk_rp.png').force_reload()


#simulation receiver page
@ui.page('/receiver_page')
def receiver_page():
    with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
        ui.html('<u>Receiver Page</u>').style('font-size: 3em; font-weight: bold;')
    """This function creates the Receiver page where the user can view outputs from the receiver."""
    ui.button('back', on_click=ui.navigate.back)
    #on this page put plots


    marker_front = ''
    marker_back = ''
    payload = ''

    for i in range(len(rx_message_binary)):
        #get the first 8 bits as the marker
        if i < 32:
            marker_front += str(rx_message_binary[i])
        elif i > len(rx_message_binary) - 32:
            marker_back += str(rx_message_binary[i])
        else:
            payload += str(rx_message_binary[i])

    with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
        ui.image('demod_media/incoming.png').force_reload()
        ui.image('demod_media/filtered.png').force_reload()
        ui.image('demod_media/final_sig.png').force_reload()
        ui.image('demod_media/Constellation.png').style('width:30%').force_reload()
        ui.label('Bit Sequence:').style('font-size: 2.5em; font-weight: bold;')

        ui.html(f'''<div style ="font-size: 2.5em; font-weight: bold; color: #D2042D;"><span style = 'color:#0072BD'>Marker</span> | <span style = 'color:black'>Message</span> | <span style = 'color:#0072BD'>Marker</span></div>''').style('text-align: center;')
        ui.html(f'''<div style ="font-size: 2.5em; font-weight: bold; color: #D2042D; text-wrap:wrap; word-break: break-all;"><span style = 'color:#0072BD'>{marker_front}</span> | <span style = 'color:black; '>{payload}</span> | <span style = 'color:#0072BD'>{marker_back}</span></div>''').style('text-align: center;')
        ui.label(f'Decoded Message: {rx_recovered_message}').style('font-size: 2.5em; font-weight: bold;')

    pass

@ui.page('/advanced_page')
def advanced_page():
    #TODO bring things over from Cesium_Sattelite/viewer
    pass

# TODO implement the control page when we are able to
#control page
@ui.page('/control_page')
def control_page():
    pass


ui.run()    
