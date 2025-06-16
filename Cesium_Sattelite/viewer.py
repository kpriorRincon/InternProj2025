# Author: Kobe Prior
# Date: June 10
# Purpose: This file provides a NiceGUI-based web interface for selecting recently launched satellites,
#          generating their TLE-based CZML data, and visualizing them in a Cesium viewer.


# import necessary libraries
from nicegui import ui, app
import os
import time
from datetime import datetime, timezone, timedelta
# function I made to get current TLE data
from get_TLE import get_up_to_date_TLE

from satellite_czml import satellite_czml
'''note ctrl click satellite_czml then comment out satellites = {} because it isn't instance specific then
at the beginning of __init__() add self.satellites = {}
at the top of the class from datetime import datetime, timedelta, timezone
also replace all instances of datetime.utcnow() with datetime.now(timezone.utc)'''
from skyfield.api import load, wgs84, EarthSatellite
saved_tles = get_up_to_date_TLE()  # get the most up to date TLE
# define the position of the transmitter and receiver
tx_pos = wgs84.latlon(39.586389, -104.828889, elevation_m=1600)  # Kobe's seat at Rincon
rx_pos = wgs84.latlon(39.748056, -105.221667, elevation_m=1600)  # Kobe's dorm
# we want to declare these globally so we can reset when needed
selected = set()
sat_buttons = {}
tles = []
count = 0
sat_for_sim = None # To start until we're ready to use it
time_crossing = None
# start of site
text_box_container = ui.column().style('order: 2; width: 80%')


def update_text_boxes(e):
    """Updates the UI to display the appropriate number of satellite selection buttons based on user input."""
    global count
    count = int(e.value)
    text_box_container.clear()

    with text_box_container:
        ui.label(
            f'Select at most {count} Satellite(s) out of the last 30 days of launches')

        with ui.row().style('width:100%'):
            # create a bunch of buttons for possible sattelites to choose from
            global selected, sat_buttons

            selected = set()
            sat_buttons.clear()  # clear any old references

            def on_sat_button_click(sat_name, button):
                global saved_tles
                if sat_name in selected:
                    selected.remove(sat_name)
                    button.props('color=primary')
                elif len(selected) < count:
                    selected.add(sat_name)
                    button.props('color=green')
                # selected appears to be working well
                print(f'currently selected:{selected}')

            count = int(e.value)
            sat_buttons = {}

            # fix late binding error
            for sat_name in saved_tles['names']:
                btn = ui.button(
                    sat_name,
                    on_click=lambda e, n=sat_name, b=None: on_sat_button_click(
                        n, sat_buttons[n])
                ).props('color=primary')
                sat_buttons[sat_name] = btn


def submit():
    """Handles the submit action: collects selected satellites' TLEs, generates CZML, writes it to a file, and navigates to the Cesium viewer page."""
    global selected
    global tles  # empties the list?
    tles.clear()
    # get the corresponding data from the selected buttons
    # print(f'currently selected after submit: {selected}')
    for names in selected:
        # build the TLE array in the format ['ISS (ZARYA)','1 25544U 98067A   21016.23305200  .00001366  00000-0  32598-4 0  9992', '2 25544  51.6457  14.3113 0000235 231.0982 239.8264 15.49297436265049']
        # get the line1 correspondin to the name
        line1 = saved_tles['line1s'][saved_tles['names'].index(names)]
        line2 = saved_tles['line2s'][saved_tles['names'].index(names)]
        tles.append([names, line1, line2])

    print(f'TLE list{tles}')  # This is your array of arrays
    # convert tle_list to czml

    # Convert to CZML
    czml_obj = satellite_czml(tles)  # this should create a new object
    czml_string = czml_obj.get_czml()
    # testing if the string is gettin appended to or not
    print(len(czml_string))
    # write this string to a file
    with open('sats.czml', 'w') as f:
        # ensure that this file is being removed from the begining
        f.truncate(0)
        f.write(czml_string)
    # navigate to the page to display
    ui.navigate.to('/Cesium_page')


ui.number(label='How many Satellites?', min=1, max=10, step=1,
          on_change=update_text_boxes).style('width: 10%')
ui.button('Submit', on_click=submit, color='positive').style('order: 3;')


@ui.page('/Cesium_page')
def Cesium_page():
    global count
    # start of Cesium page

    def back_and_clear():
        global selected, sat_buttons, tles
        for btn in sat_buttons.values():
            btn.props('color=primary')
        selected.clear()
        ui.navigate.back()
    ui.button('Back', on_click=back_and_clear)
    # TODO find the time when the first satellite crosses the line of site of both ground stations e.g. tx_pos and rx_pos
    # create the satellites
    # note that the tles list is a list of lists so we can get each list one by one and take the corresponding data to build EarthSatellite objects
    satellites = [EarthSatellite(tle[1], tle[2], tle[0]) for tle in tles]
    # set the distance that the satellite must be from both ground stations
    thresh_km = 2000
    # define start time and range
    ts = load.timescale()
    now_utc = datetime.now(timezone.utc)
    start_time = ts.utc(now_utc)  # convert to skyfield time
    # do in 3 hour steps
    # Increase frequency: check every 5 minutes (0.0833 hours)
    # Check every minute for 2 days: 2 days * 24 hours * 60 minutes = 2880 steps
    time_range = [start_time + timedelta(minutes=i)
                  for i in range(2 * 24 * 60)]
    # Track unique satellite crossings, only keep minimum distance per satellite we can do the number of satellites the user selected as the cap
    crossings = {}

    '''For loops explanation:
        # For each time step in the 2-day range (1-minute increments), check every satellite's distance to both ground stations.
        # If a satellite is within the threshold distance to both, record its closest approach time and distances.
        # Stop searching once the required number of unique satellite crossings is found.'''
    for t in time_range:
        for sat in satellites:
            uplink_dist = (sat - tx_pos).at(t).distance().km
            downlink_dist = (sat - rx_pos).at(t).distance().km
            if uplink_dist < thresh_km and downlink_dist < thresh_km:
                sat_name = sat.name
                min_dist = min(uplink_dist, downlink_dist)
                # Only keep the closest approach for each satellite
                if sat_name not in crossings or min_dist < crossings[sat_name]['min_dist']:
                    crossings[sat_name] = {
                        'time': t.utc_strftime('%Y-%m-%d %H:%M:%S'),
                        'uplink_dist': uplink_dist,
                        'downlink_dist': downlink_dist,
                        'min_dist': min_dist
                    }
        if len(crossings) >= count:
            break
    with ui.column().style('width: 25%'):
        ui.label('Closest Satellite Crossings (1 minute increments for 2 days)').style(
            'font-size: 1.5em; font-weight: bold; margin-top: 1em; margin-bottom: 0.5em; white-space: normal; word-break: break-word;')


        def store_sat(name, data):
            #when the user clicks on one of the rows get information needed for the simulation and store them in 
            global time_crossing, sat_for_sim 
            # get the satellite we want
            for sat in satellites: 
                if sat.name == name:
                    sat_for_sim = sat
                    break
            time_crossing = data['time']#extracted from data
            ui.navigate.to('/simulation_page')
            return
        
        # Print the first 5 unique satellite crossings with their minimum distances

        # description of for loop you count while you go through the crossing list where sat_name is the key in the dictionary and data is 'time', 'uplink_dist', ...
        for i, (sat_name, data) in enumerate(list(crossings.items())[:count], 1):
            print(f"{i}. Satellite '{sat_name}' closest approach at {data['time']} UTC")
            print(f"   Distance to Tx: {data['uplink_dist']:.1f} km")
            print(f"   Distance to Rx: {data['downlink_dist']:.1f} km")
            with ui.row().classes(
                'bg-gray-400 rounded-lg mb-2 px-4 py-2 cursor-pointer transition hover:bg-green-400'
                ).on('click', lambda e, n=sat_name, d=data:(store_sat(n,d), ui.navigate.to('/simulation_page'))):
                
                ui.label(f"{i}. Satellite '{sat_name}' closest approach at {data['time']} UTC")
                ui.label(f"   Distance to Tx: {data['uplink_dist']:.1f} km")
                ui.label(f"   Distance to Rx: {data['downlink_dist']:.1f} km")
        
        # get this files working directory
        html_directory = os.path.dirname(__file__)
        # add the files available
        app.add_static_files('/static', html_directory)

    ui.html(
        f'''
        <div style="position: fixed; top: 0; right: 0; width: 70vw; height: 95vh; border: none; margin: 1vh 1vw 0 0; padding: 0; overflow: hidden; z-index: 999999; box-shadow: 0 0 10px rgba(0,0,0,0.2); background: #fff; border-radius: 12px;">
            <iframe style="width: 100%; height: 100%; border: none;" src="/static/viewer.html?t={int(time.time())}"></iframe>
        </div>
        '''
    )
    @ui.page('/simulation_page')
    def simulation_page(): 
        # we will navigate to here whenever a row is clicked
        global time_crossing, sat_for_sim
        #for now just create a label that prints out the parameters that will be used in the simulation
        dt_crossing = datetime.strptime(time_crossing, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        time_crossing_skyfield = ts.from_datetime(dt_crossing)
        geocentric = sat_for_sim.at(time_crossing_skyfield)
        position_vector = geocentric.position.km
        velocity_vector = geocentric.velocity.km_per_s

        ui.label(f'Satellite Position Vector: {position_vector}').style('font-size: 1.5em; font-weight: bold;')
        ui.label(f'Satellite Velocity Vector: {velocity_vector}')
        ui.image('../Phys_Sim/media/doppler_eqn.png')


ui.run()
