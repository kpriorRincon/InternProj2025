from nicegui import ui, app
import os
import time
import pickle

selected = set()
# Load TLE data from pickle file if it exists
tle_pickle_path = os.path.join(os.path.dirname(__file__), 'sattelite_tles.pkl')
if os.path.exists(tle_pickle_path):
    with open(tle_pickle_path, 'rb') as f:
        saved_tles = pickle.load(f)
else:
    saved_tles = []



text_box_container = ui.column().style('order: 2; width: 70%')
inputs = []

def update_text_boxes(e):
    global inputs
    count = int(e.value)
    text_box_container.clear()
    inputs.clear()

    with text_box_container:
        ui.label('Insert the TLE data for Satellites')
        ui.link('Get TLE\'s', target='https://orbit.ing-now.com/low-earth-orbit/', new_tab=True)

        with ui.row().style('width:80%'):
            #create a bunch of buttons for possible sattelites to choose from
            global selected
            selected = set()
            def on_sat_button_click(sat_name, button):
                if sat_name in selected:
                    selected.remove(sat_name)
                    button.props('color=primary')
                elif len(selected) < count:
                    selected.add(sat_name)
                    button.props('color=green')
                # Prevent selecting more than count satellites
                else:
                    return

            count = int(e.value)
            sat_buttons = {}
            for sat_name in saved_tles['names']:
                btn = ui.button(
                    sat_name,
                    on_click=lambda e, n=sat_name, b=None: on_sat_button_click(n, sat_buttons[n])
                ).props('color=primary')
                sat_buttons[sat_name] = btn
                

def submit():
    tle_list = []
    #get the corresponding data from the selected buttons 
    for names in selected:
            #build the TLE array in the format ['ISS (ZARYA)','1 25544U 98067A   21016.23305200  .00001366  00000-0  32598-4 0  9992', '2 25544  51.6457  14.3113 0000235 231.0982 239.8264 15.49297436265049']
            line1 = saved_tles['line1s'][saved_tles['names'].index(names)] # get the line1 correspondin to the name
            line2 = saved_tles['line2s'][saved_tles['names'].index(names)]
            tle_list.append([names, line1,line2])
            
            #debug
            print(tle_list)
    print(tle_list)  # This is your array of arrays
    #convert tle_list to czml 
    from satellite_czml import satellite_czml
    # Convert to CZML
    czml_string = satellite_czml(tle_list=tle_list).get_czml()

    #add time stamp 
    timestamp = int(time.time())
    #write this string to a file
    with open('sats.czml', 'w') as f:
        f.write(czml_string)

    #clear input
    inputs.clear()
    #navigate to the page to display
    ui.navigate.to('/Cesium_page')

ui.number(label='How many Satellites?', min=0, max=5, step=1, on_change=update_text_boxes).style('width: 10%')
ui.button('Submit', on_click=submit).style('order: 3;')

@ui.page('/Cesium_page')
def Cesium_page():
    html_directory = os.path.dirname(__file__)#get this files working directory
    app.add_static_files('/static', html_directory)#add the files available
    #added ?t=time.time for cache busting
    ui.html(
        f'''
        <div style="position: fixed; top: 0; right: 0; width: 70vw; height: 95vh; border: none; margin: 1vh 1vw 0 0; padding: 0; overflow: hidden; z-index: 999999; box-shadow: 0 0 10px rgba(0,0,0,0.2); background: #fff; border-radius: 12px;">
            <iframe style="width: 100%; height: 100%; border: none;" src="/static/viewer.html?t={int(time.time())}"></iframe>
        </div>
        '''
    )

ui.run()
