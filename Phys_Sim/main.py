# this file will be used to run the GUI
from nicegui import ui
import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater

with ui.row():
    simulate_button = ui.image('/media/simulate_low_opacity.png')
    #command_button = ui.image('Phys_Sim/media/command_low_opacity.png', width='40%')

ui.run()