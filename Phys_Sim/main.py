# this file will be used to run the GUI
from nicegui import ui
import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater

# global objects for the Sig_Gen, Receiver, and Repeater classes
sig_gen = Sig_Gen.SigGen()
repeater = Repeater.Repeater(desired_frequency=915e6, sampling_frequency=1e6)
receiver = Receiver.Receiver(sampling_rate=1e6, frequency=915e6)



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

 #   Simulate page
    
    simulation_container = ui.column().style('order: 2;')

    #this button needs to have a lambda function that will fill all the variables in the Sig_Gen, Receiver, and Repeater classes
    #TODO drop down menu to select the type of simulation
    with ui.row().style('justify-content: center;'):
        ui.label('Simulation Type').style('font-size: 2em; font-weight: bold;')
    def open_simulation_single_message():
        selected_type = simulation_type_dropdown.value
        simulation_container.clear()
        if selected_type == 'Single Message':
            with simulation_container:
                ui.label('Single Message Simulation').style('font-size: 2em; font-weight: bold; ')
                ui.label('Enter message to be sent')
                message_input = ui.input(placeholder="hello world")
                ui.label('Simulation Parameters').style('font-size: 2em; font-weight: bold;')
                # When the user selects a simulation type, the parameters will change accordingly
                ui.label('Frequency In (MHz)').style('width: 200px; margin-bottom: 10px;')
                freq_in_slider = ui.slider(min=902, max=928, step=1).props('label-always')
                ui.label('Frequency Out (MHz)').style('width: 200px; margin-bottom: 10px;')
                freq_out_slider = ui.slider(min=902, max=928, step=1).props('label-always')
                ui.label('Gain (dB)').style('width: 200px;margin-bottom: 10px;')
                gain_slider = ui.slider(min=0, max=10, step=1).props('label-always')
                ui.label('Noise Level (dB)').style('width: 200px; margin-bottom: 10px;')
                noise_slider = ui.slider(min=0, max=10, step=1).props('label-always')
                #TODO the submit button will be a handler that stores are variable data for the Sig_Gen, Receiver, and Repeater classes
                ui.button("Submit", on_click=lambda: store_data()).style('width: 200px; height: 10px;')
                def store_data():
                    """
                    store_data()
                    This function is triggered when the user clicks the submit button.
                    It collects the values from the input fields and stores them for further processing.
                    """
                    # Here you would typically collect the values from the input fields and store them
                    # For example:
                    global sig_gen
                    global receiver
                    global repeater
                    sig_gen.freq = int(freq_in_slider.value)* 1e6  # Convert MHz to Hz
                    sig_gen.sample_rate = 20 * sig_gen.freq  # Example sample rate 20 times the frequency
                    repeater.desired_freqeuncy = int(freq_out_slider.value)
                    repeater.sampling_fequency = int(sig_gen.sample_rate)
                    repeater.gain = 10^(int(gain_slider.value)/10)
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

#control page
@ui.page('/control_page')
def control_page():
    pass


ui.run()    
# with ui.row().style('justify-content: flex-start;'):