# this file will be used to run the GUI
from nicegui import ui
import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater

# global objects for the Sig_Gen, Receiver, and Repeater classes
sig_gen = Sig_Gen.SigGen()
repeater = Repeater.Repeater(desired_frequency=915e6, sampling_frequency=1e6, gain=1)
receiver = Receiver.Receiver(sampling_rate=1e6, frequency=915e6)
noise_bool = False  # Global variable to control noise addition
noise_power = 0.1  # Default noise power


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
                message_input = ui.input(placeholder="hello world")
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
                    store_data()
                    This function is triggered when the user clicks the submit button.
                    It collects the values from the input fields and stores them for further processing.
                    """
                    #using global to change attributes of the global objects
                    global sig_gen
                    global receiver
                    global repeater
                    sig_gen.freq = int(freq_in_slider.value)* 1e6  # Convert MHz to Hz
                    sig_gen.sample_rate = 20 * sig_gen.freq  # Example sample rate 20 times the frequency
                    message = message_input.value
                    sig_gen.message_to_bits(message) # note this appends prefixes with 'R'
                    repeater.desired_freqeuncy = int(freq_out_slider.value)
                    repeater.sampling_fequency = int(sig_gen.sample_rate)
                    repeater.gain = 10**(int(gain_slider.value)/10) # convert dB to linear scale
                    if noise_checkbox.value:
                        global noise_bool
                        noise_bool = True
                        global noise_power
                        noise_power = 10**(int(noise_slider.value)/10)  # Convert dB to linear scale
                    else:
                        noise_bool = False
                        noise_power = 0  # Default value if no noise is added

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
    pass
#simulation Repeater page
@ui.page('/repeater_page')
def repeater_page():
    """This function creates the repeater page where the user can view outputs from the repeater."""
    ui.button('back', on_click=ui.navigate.back)

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
# with ui.row().style('justify-content: flex-start;'):