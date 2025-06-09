from nicegui import ui, app
import os

#start here 
# The user would be starting here where they enter a series of TLE codes, up to 5
text_box_container = ui.column().style('order: 2; width: 70%')
inputs = []
def update_text_boxes(e):
    #start by clearin gthe simulation container
    #based on e we allocate a number of text boxes
    global inputs
    count = int(e.value)
    text_box_container.clear()
    inputs.clear()
    
    with text_box_container:
        ui.label('Insert the TLE data for Satellites')
        ui.link('Get TLE\'s', target = 'https://orbit.ing-now.com/low-earth-orbit/', new_tab=True)
        for i in range(count):
            with ui.row():
                name = ui.input(label=f'Satellite {i + 1} Name')
                line1 = ui.input(label=f'Satellite {i + 1} TLE Line 1')
                line2 = ui.input(label=f'Satellite {i + 1} TLE Line 2')
            inputs.append((name, line1, line2))
def submit():
    from satellite_czml import satellite_czml
    sats = [[name.value, line1.value, line2.value] for name, line1, line2 in inputs]
    print(sats)
    #use the sat values and create a czml file
    # czml_string = satellite_czml(tle_list=sats).get_czml()
    # #write this string to a file
    # with open('sats.czml', 'w') as f:
    #     f.write(czml_string)
    # ui.notify(f'satellites: {sats}')  
    # #once everything is ready we can go to the cesium page
    # ui.navigate.to('/Cesium_page')

ui.number(label='How many Satellites?', min=0, max=5, step=1, on_change=update_text_boxes).style('width: 10%')
ui.button('Submit', on_click = submit).style('order: 3;')


@ui.page('/Cesium_page')

def Cesium_page():

    html_directory = os.path.dirname(__file__)#get the directory of the current script
    app.add_static_files('/static', html_directory) # serve the script
    #next steps I want the user to be able to copy paste several TLE's for satellites 
    with ui.row().style('width: 100%; height: 100vh'):
        with ui.column().style('width: 90%'):
            ui.html('<iframe src=/static/viewer.html></iframe>').style('width: 100%; height: 90vh; border: none;')



ui.run()