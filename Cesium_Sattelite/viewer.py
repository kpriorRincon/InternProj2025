# Author: Kobe Prior
# Date: June 10
# Purpose: This file provides a NiceGUI-based web interface for selecting recently launched satellites,
#          generating their TLE-based CZML data, and visualizing them in a Cesium viewer.

# import necessary libraries
from nicegui import ui, app
import os
import time
import pickle
from get_TLE import get_up_to_date_TLE
from satellite_czml import satellite_czml 
#note ctrl click satellite_czml then comment out satellites = {} because it isn't instance specific then 
#at the beginning of __init__() add self.satellites = {}
#at the top of the class from datetime import datetime, timedelta, timezone
#also replace all instances of datetime.utcnow() with datetime.now(timezone.utc)
if __name__ == "__main__":
    #only do this once since nicegui uses multiprocessing
    saved_tles = get_up_to_date_TLE() #get the most up to date TLE 

#we want to declare these globally so we can reset when needed
selected = set()
sat_buttons  ={} 

#start of site
text_box_container = ui.column().style('order: 2; width: 80%')


def update_text_boxes(e):
    count = int(e.value)
    text_box_container.clear()

    with text_box_container:
        ui.label(f'Select {count} Satellite(s) out of the last 30 days of launches')

        with ui.row().style('width:100%'):
            #create a bunch of buttons for possible sattelites to choose from
            global selected, sat_buttons

            selected = set()
            sat_buttons.clear()#clear any old references
            def on_sat_button_click(sat_name, button):
                if sat_name in selected:
                    selected.remove(sat_name)
                    button.props('color=primary')
                elif len(selected) < count:
                    selected.add(sat_name)
                    button.props('color=green')
                print(f'currently selected:{selected}') #selected appears to be working well

            count = int(e.value)
            sat_buttons = {}

            #fix late binding error
            for sat_name in saved_tles['names']:
                btn = ui.button(
                    sat_name,
                    on_click=lambda e, n=sat_name, b=None: on_sat_button_click(n, sat_buttons[n])
                ).props('color=primary')
                sat_buttons[sat_name] = btn
                

def submit():
    global selected
    tles = []#empties the list?
    #get the corresponding data from the selected buttons 
    #print(f'currently selected after submit: {selected}')
    for names in selected:
            #build the TLE array in the format ['ISS (ZARYA)','1 25544U 98067A   21016.23305200  .00001366  00000-0  32598-4 0  9992', '2 25544  51.6457  14.3113 0000235 231.0982 239.8264 15.49297436265049']
            line1 = saved_tles['line1s'][saved_tles['names'].index(names)] # get the line1 correspondin to the name
            line2 = saved_tles['line2s'][saved_tles['names'].index(names)]
            tles.append([names,line1,line2])
            
    print(f'TLE list{tles}')  # This is your array of arrays
    #convert tle_list to czml 

    # Convert to CZML
    czml_obj = satellite_czml(tles)#this should create a new object
    czml_string = czml_obj.get_czml()
    print(len(czml_string))#testing if the string is gettin appended to or not
    #write this string to a file
    with open('sats.czml', 'w') as f:
        f.truncate(0)#ensure that this file is being removed from the begining 
        f.write(czml_string)
    #navigate to the page to display
    ui.navigate.to('/Cesium_page')

ui.number(label='How many Satellites?', min=1, max=10, step=1, on_change=update_text_boxes).style('width: 10%')
ui.button('Submit', on_click=submit, color='positive').style('order: 3;')

@ui.page('/Cesium_page')
def Cesium_page():
    def back_and_clear():
        global selected, sat_buttons
        for btn in sat_buttons.values():
            btn.props('color=primary')
        selected.clear()
        ui.navigate.back()
    ui.button('Back', on_click=back_and_clear)
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
