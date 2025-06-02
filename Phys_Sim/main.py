# this file will be used to run the GUI
from nicegui import ui
import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater


#front page
with ui.row().style('height: 100vh; width: 100%; display: flex; justify-content: center; align-items: center;'):
    with ui.link(target='youtube.com'):
        simulate_button = ui.image('media/simulate.png').style('width: 50vh; height: 50vh;')
    #TODO: make this command and control the opacity of the image
    #simulate_copy = ui.image('media/simulate_low_opacity.png').style('width: 50vh; height: 50vh;')

ui.run()
# with ui.row().style('justify-content: flex-start;'):