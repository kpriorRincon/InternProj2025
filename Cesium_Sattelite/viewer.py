from nicegui import ui, app
import os
html_directory = os.path.dirname(__file__)#get the directory of the current script
app.add_static_files('/static', html_directory) # serve the script
with open('viewer.html', 'r') as f:
    cesium_html = f.read()

#start here 
# The user would be starting here where they enter a series of TLE codes, up to 5
text_box_container = ui.column().style('order: 2;')
inputs= []
def update_text_boxes(e):
    #start by clearin gthe simulation container
    #based on e we allocate a number of text boxes
    count = int(e.value)
    text_box_container.clear()
    inputs.clear()
    
    with text_box_container:
        for i in range(count):
            tb = ui.input(label=f'Satellite {i + 1}', placeholder=f'Enter name {i + 1}')
            inputs.append(tb)
def submit():
    sats = [tb.value for tb in inputs]
    ui.notify(f'satellites: P{sats}')  

ui.number(label='How many Satellites?', min=0, max=5, step=1, on_change=update_text_boxes).style('width: 10%')
ui.button('Submit', on_click = submit)


def store_TLE():
    #when the submit function is pressed get all the data that was entered 
    pass

@ui.page('Cesium_page')

def Cesium_page():

#Have one page where the user enteres the TLE data 
#next steps I want the user to be able to copy paste several TLE's for satellites 
    with ui.row().style('width: 100%; height: 100vh'):
        #left
        with ui.column().style('width: 30%;'):
            ui.label('nothing')
        #right
        with ui.column():
            ui.label('nothing')
            ui.html('<iframe src=/static/viewer.html></iframe>').force_reload()


ui.run()