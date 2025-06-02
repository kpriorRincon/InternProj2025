# this file will be used to run the GUI
from nicegui import ui
import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater


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
                message_input = ui.input(on_change=lambda e: ui.notify(f'Message entered: {e.value}'))
                ui.label('Simulation Parameters').style('font-size: 2em; font-weight: bold;')
                # When the user selects a simulation type, the parameters will change accordingly
                ui.label('Frequency In (Hz)').style('width: 200px;')
                #freq_in_slider = ui.slider()
                ui.label('Frequency Out (Hz)').style('width: 200px;')
                # freq_out_slider = ui.slider()
                ui.label('Gain (dB)').style('width: 200px;')
                # gain_slider = ui.slider()
                ui.label('Noise Level (dB)').style('width: 200px;')
                # noise_slider = ui.slider()

                ui.button("Submit", on_click=lambda: ui.notify('Simulation started!')).style('width: 200px; height: 10px;')
            
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