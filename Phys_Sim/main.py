#TODO insert file header here
# this file will be used to run the GUI
from nicegui import ui
import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import time
# global objects for the Sig_Gen, Receiver, and Repeater classes
symbol_rate = 10
sample_rate = 4e9
sig_gen = Sig_Gen.SigGen(sample_rate = sample_rate, symbol_rate = symbol_rate)
repeater = Repeater.Repeater(sampling_frequency=sample_rate)
receiver = Receiver.Receiver(sampling_rate=sample_rate)


noise_bool = False  # Global variable to control noise addition
noise_power = 0.1  # Default noise power
message_input = None  # Global variable to store the message input field



#front page
with ui.row().style('height: 100vh; width: 100%; display: flex; justify-content: center; align-items: center;'):
    with ui.link(target='\simulate_page'):
        simulate_button = ui.image('media/simulate.png').style('width: 50vh; height: 50vh;')
    #TODO: make this command and control the opacity of the image
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
                message = ui.input(placeholder="hello world")
                ui.label('Simulation Parameters').style('font-size: 2em; font-weight: bold;')
                # When the user selects a simulation type, the parameters will change accordingly
                ui.label('Frequency In (MHz)').style('width: 200px; margin-bottom: 10px;')
                freq_in_slider = ui.slider(min=902, max=928, step=1).props('label-always')
                ui.label('Frequency Out (MHz)').style('width: 200px; margin-bottom: 10px;')
                freq_out_slider = ui.slider(min=902, max=928, step=1).props('label-always')
                ui.label('Gain (dB)').style('width: 200px;margin-bottom: 10px;')
                gain_slider = ui.slider(min=0, max=10, step=1).props('label-always')
                #check box to ask if the user wants to add noise
                ui.label('Add Noise?').style('width: 200px;')
                noise_checkbox = ui.checkbox('add noise')#the value here will be a bool that can be used for siGen
                #iff the user checks the noise checkbox, then show the noise slider

                ui.label('Noise Level (dB)').style('width: 200px; margin-bottom: 10px;').bind_visibility_from(noise_checkbox, 'value')
                noise_slider = ui.slider(min=0, max=10, step=1).props('label-always').bind_visibility_from(noise_checkbox, 'value')


                ui.button("Submit", on_click=lambda: store_data()).style('width: 200px; height: 10px;')
                def store_data():
                    """
                    stoample_rate = sample_rate, symbol_rate = symbol_ratee_data()
                    This function is triggered when the user clicks the submit button.
                    It collects the values from the input fields and stores them for further processing.
                    """
                    #using global to change attributes of the global objects
                    global sig_gen
                    global receiver
                    global repeater
                    global message_input
                    message_input = message.value
                    #run the sig gen handler
                    sig_gen.handler(message.value, int(freq_in_slider.value)*1e6) 

                    repeater.desired_freqeuncy = int(freq_out_slider.value) * 1e6
                    #repeater.sampling_fequency = int(sig_gen.sample_rate)
                    repeater.gain = 10**(int(gain_slider.value)/10) # convert dB to linear scale
                    #add receiver things as well
                    repeater.handler(sig_gen.time_vector, sig_gen.qpsk_waveform, sig_gen.freq)

                    if noise_checkbox.value:
                        global noise_bool
                        noise_bool = True
                        global noise_power
                        noise_power = 10**(int(noise_slider.value)/10)  # Convert dB to linear scale
                    else:
                        noise_bool = False
                        noise_power = 0  # Default value if no noise is added

                    #TODO put receiver class here
                    _,best_bits = receiver.demodulator(sig_gen.qpsk_waveform, sig_gen.sample_rate, sig_gen.symbol_rate, repeater.desired_freqeuncy)
                    receiver.frequency=repeater.desired_freqeuncy

                    #noise_level = noise_slider.value
                    #debug:
                    #print("made it here")
                    ui.notify('Data stored successfully!')  # Placeholder notification

            
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
    global sig_gen 
    if message_input is not None:
        #time stamp to force refresh
        time_stamp = int(time.time())
        with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
            ui.label(f'Message entered: {message_input}').style('font-size: 2em; font-weight: bold;')
            #we want to show the header in a different color as the actual message 
            bit_sequence =sig_gen.message_to_bits(message_input)
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
    ui.image('repeater.png')
    pass

#simulation receiver page
@ui.page('/receiver_page')
def receiver_page():
    """This function creates the Receiver page where the user can view outputs from the receiver."""
    ui.button('back', on_click=ui.navigate.back)

    pass



# TODO implement the control page when we are able to
#control page
@ui.page('/control_page')
def control_page():
    pass


ui.run()    
