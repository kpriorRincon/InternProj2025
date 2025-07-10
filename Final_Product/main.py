"""
Rincon Research Internship Project Red Mountain

Authors: Skylar Harris, Jorge Hernandez, Kobe Prior, and Trevor Wiseman
Date: July 2nd, 2025

File Description:
This file, when run, creates a graphical user interface (GUI) that offers the user two main options:
1. Simulate a bent pipe communication system involving a satellite and two ground stations.
2. Command and control hardware to send messages in a similar manner at a smaller scale.
"""
# Author: Kobe Prior
# Date: June 10
# Purpose: This file provides a NiceGUI-based web interface for selecting recently launched satellites,
#          generating their TLE-based CZML data, and visualizing them in a Cesium viewer.

#a comment to test push from vm 
# import necessary libraries
from nicegui import ui, app
import os
import time
import zmq, zmq.asyncio
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import asyncssh


from datetime import datetime, timezone, timedelta
# function I made to get current TLE data
from get_TLE import get_up_to_date_TLE
# importing classes
import Sig_Gen as SigGen
import Channel as Channel
from config import *
from binary_search_caf import channel_handler
from satellite_czml import satellite_czml # See Readme for mor information this class must be modified
#downgrade pygeoif: pip install pygeoif==0.7.0
from skyfield.api import load, wgs84, EarthSatellite
import pathlib

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
recovered_message = None # this will be set in the simulation page when we recover the message
required_rep_power = None # this will be set in the simulation page when we calculate the required repeater power  
txFreq = None
bits = None
#matplotlib rc
plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 16,      # X and Y axis label size
    'xtick.labelsize': 14,     # X tick label size
    'ytick.labelsize': 14,     # Y tick label size
    'legend.fontsize': 14,     # Legend font size
    'figure.titlesize': 22     # Figure suptitle size
})
# get this files working directory
html_directory = os.path.dirname(__file__) #get the directory you're in
# add the files available
app.add_static_files('/static', html_directory)

def inject_head_style(page_name):
    with open('html_head.html') as f:
        head =f"<script> document.title = '{page_name}';"
        head = "'''" + head + f.read()
        ui.add_head_html(head)
# Enhanced CSS for hover zoom effect in flex containers
ui.add_css('''
.thumbnail {
    width: 350px;
    position: relative;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    cursor: pointer;
    z-index: 1;
    border-radius: 8px; /* optional: rounded corners */
}
.thumbnail:hover {
    transform: scale(2.0);
    z-index: 10;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

.flex-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 20px;
    width: 100%;
    padding: 200px; /* add some padding to prevent edge clipping */
}
''', shared=True)

def zoomable_image(src):
    # input file path to the image
    # returns an image that can be zoomed in on hover
    return ui.image(src).classes('thumbnail').force_reload()


#Front page 
#Add a cool background image
ui.add_head_html('''
    <style>
    html, body {
        margin: 0;
        padding: 0;
        width: 100vw;
        min-width: 100vw;
        box-sizing: border-box;
    }
    body {
        background-image: url("https://adventr.co/wp-content/uploads/2019/08/Red2Cover.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat; 
        margin: 0;
        padding: 0;
        width: 100vw;
        min-width: 100vw;
        box-sizing: border-box;
    }
    
    .glass-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        min-width: 100vw;
        height: 60px;
        display: flex;
        gap: 20px;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.1);
        -webkit-backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        z-index: 1000;
        padding: 0 30px;
        box-sizing: border-box;
        color: white;
    }

    .glass-bar .item {
        display: flex;
        align-items: center;
        cursor: pointer;
        gap: 8px;
        font-size: 18px;
        color: white;
        transition: background 0.2s;
    }

    .glass-bar .item:hover {
        scale: 1.05;
    }

    .spacer {
        height: 60px; /* reserve space under the fixed bar */
    }
                 
    .card-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 32px;
        min-height: 32vh;
        width: 70vw;
        box-sizing: border-box;
        margin: 0 auto;
    }

    .card {
        background: rgba(255,255,255,0.65);
        border-radius: 10px;
        padding: 32px 24px;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        cursor: pointer;
        box-sizing: border-box;
    }
    .card label {
        background: rgba(255,255,255,0.92) !important;
        border-radius: 6px;
        padding: 6px 10px;
        margin-bottom: 6px;
        display: block;
        width: 100%;
        box-sizing: border-box;
    }
    .card:hover {
        scale: 1.05;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 110vh;
    width: 100vw;
    box-sizing: border-box;
}
    </style>
''')

#Add the frosted glass bar
with ui.element('div').classes('glass-bar'):
    with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/')):
        ui.icon('home')
        ui.label('Home')
    with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/SIMULATE')):
        ui.icon('code')
        ui.label('Simulation')

    with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/CONTROL')):
        ui.icon('settings')
        ui.label('Control')

    with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/ABOUT')):
        ui.icon('info')
        ui.label('About')
#add some spacer to content doesn't go under the fixed bar
ui.element('div').classes('spacer')

#Main page content:
with ui.element('div').classes('wrapper'):
    with ui.element('div').classes('card-container'):
        with ui.element('div').classes('card').on('click', lambda: ui.navigate.to('/SIMULATE')):
            ui.image('media/simulate.png').force_reload()  # Ensure the image is always fresh
            ui.label('Simulate a bent pipe communication system with a satellite and two ground stations.').style('text-align: center; font-size: 1.3em;')

        with ui.element('div').classes('card').on('click', lambda: ui.navigate.to('/CONTROL')):
            ui.image('media/control.png').force_reload()  # Ensure the image is always fresh
            ui.label('Control software defined radios to send a message to a transponder and receive it back.').style('text-align: center;font-size: 1.3em;')

        with ui.element('div').classes('card').on('click', lambda: ui.navigate.to('/ABOUT')):
            ui.image('media/about.png').force_reload()  # Ensure the image is always fresh
            ui.label('Learn more about the project and its authors.').style('text-align: center;font-size: 1.3em;')

# Add a footer with a link to the GitHub repository


# Simulate Page-------------------------------------------------------------------------------------------------------------------
@ui.page('/SIMULATE')
def simulate_page():
    inject_head_style('simulate')
    with ui.element('div').classes('glass-bar'):
        with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/')):
            ui.icon('home')
            ui.label('Home')
        with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/SIMULATE')):
            ui.icon('code')
            ui.label('Simulation')

        with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/CONTROL')):
            ui.icon('settings')
            ui.label('Control')

        with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/ABOUT')):
            ui.icon('info')
            ui.label('About')
    #add some spacer to content doesn't go under the fixed bar
    ui.element('div').classes('spacer')

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
        global tles  
        # empties the list
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
        czml_obj = satellite_czml(tles)  # this should create a new object and with the modifications at the top of the program they each instance will have it's own dictionary. 
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


    with ui.row().style('justify-content: center; align-items: center; width: 100%; margin-top: 2em;'):
        ui.number(
            label='How many Satellites?', 
            min=1, 
            max=10, 
            step=1, 
            on_change=update_text_boxes
        ).style('width: 18%; font-size: 1.3em;')
        ui.button('Submit', on_click=submit, color='positive').style('margin-left: 2em; font-size: 1.1em;')


    @ui.page('/Cesium_page')
    def Cesium_page():
        inject_head_style('Cesium')
        with ui.element('div').classes('glass-bar'):
            with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/')):
                ui.icon('home')
                ui.label('Home')
            with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/SIMULATE')):
                ui.icon('code')
                ui.label('Simulation')

            with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/CONTROL')):
                ui.icon('settings')
                ui.label('Control')

            with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/ABOUT')):
                ui.icon('info')
                ui.label('About')
        #add some spacer to content doesn't go under the fixed bar
        ui.element('div').classes('spacer')

        global count, tx_pos, rx_pos
        # start of Cesium page

        def back_and_clear():
            global selected, sat_buttons, tles
            for btn in sat_buttons.values():
                btn.props('color=primary')
            selected.clear()
            ui.navigate.back()
        ui.button('Back', on_click=back_and_clear)


        #for each of the tles create a list satellite and store it in an array
        # note that the tles list is a list of lists so we can get each list one by one and take the corresponding data to build EarthSatellite objects
        satellites = [EarthSatellite(tle[1], tle[2], tle[0]) for tle in tles]
        # set the distance that the satellite must be from both ground stations
        thresh_km = 2000

        # define start time and range
        ts = load.timescale()
        now_utc = datetime.now(timezone.utc)
        start_time = ts.utc(now_utc)  # convert to skyfield time

        # Check every minute for 1 day: 1 day * 24 hours * 60 minutes = 1440 steps
        time_range = [start_time + timedelta(minutes=i) for i in range(1 * 24 * 60)]

        # Track unique satellite crossings, only keep minimum distance per satellite we can do the number of satellites the user selected as the cap
        crossings = {}

        '''For loops explanation:
            # For each time step in the 1-day range (1-minute increments), check every satellite's distance to both ground stations.
            # If a satellite is within the threshold distance to both, record its closest approach time and distances.
            # Stop searching when you've finished the 1 day time range'''
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
        with ui.column().style('width: 25%'):
            ui.label('Closest Satellite Crossings (1 minute increments for 1 day)').style(
                'font-size: 1.5em; font-weight: bold; margin-top: 1em; margin-bottom: 0.5em; white-space: normal; word-break: break-word;')


            def store_sat(name, data):
                #when the user clicks on one of the rows get information needed for the simulation and store them in global variables 
                global time_crossing, sat_for_sim 
                # get the satellite we want
                for sat in satellites: 
                    if sat.name == name:
                        sat_for_sim = sat
                        break
                time_crossing = data['time'] #get the time of closest approach 
                ui.navigate.to('/simulation_page')
                return
            
            '''description of for loop:
            count while you go through the crossing list 
            where sat_name is the key in the dictionary and data is 'time', 'uplink_dist', ...
            ''' 
            for i, (sat_name, data) in enumerate(list(crossings.items())[:count], 1):
                print(f"{i}. Satellite '{sat_name}' closest approach at {data['time']} UTC")
                print(f"   Distance to Tx: {data['uplink_dist']:.1f} km")
                print(f"   Distance to Rx: {data['downlink_dist']:.1f} km")
                with ui.row().classes(
                    'bg-gray-400 rounded-lg mb-2 px-6 py-2 cursor-pointer transition hover:bg-green-400'
                    ).on('click', lambda e, n=sat_name, d=data:(store_sat(n,d), ui.navigate.to('/simulation_page'))):
                    #the row consists of information about the satellite name the time of crossin gand the distance to uplink and downlink
                    ui.html(f"""
                        <div style='margin-bottom: 1em;'>
                            <div style='font-size: 1.1em; font-weight: bold; color: #1a237e;'>
                                {i}. Satellite '{sat_name}' closest approach at
                            </div>
                            <div style='font-size: 1.3em; font-weight: bold;'>
                                {data['time']} UTC
                            </div>
                            <div style='font-size: 1.1em; font-weight: bold; color: #1a237e;'>
                                Distance to Tx: {data['uplink_dist']:.1f} km
                            </div>
                            <div style='font-size: 1.1em; font-weight: bold; color: #1a237e;'>
                                Distance to Rx: {data['downlink_dist']:.1f} km
                            </div>
                        </div>
                    """)
            

        #cesium page take up the right 70 percent of the page
        ui.html(
            f'''
            <div style="position: absolute; top: 70px; right: 0; width: 70vw; height: 90vh; border: none; margin: 1vh 1vw 0 0; padding: 0; overflow: hidden; z-index: 999999; box-shadow: 0 0 10px rgba(0,0,0,0.2); background: #fff; border-radius: 12px;">
            <iframe style="width: 100%; height: 100%; border: none;" src="/static/viewer.html?t={int(time.time())}"></iframe>
            </div>
            '''
        )

        @ui.page('/simulation_page')
        def simulation_page(): 
            inject_head_style('simulation_page')
            with ui.element('div').classes('glass-bar'):
                with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/')):
                    ui.icon('home')
                    ui.label('Home')
                with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/SIMULATE')):
                    ui.icon('code')
                    ui.label('Simulation')

                with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/CONTROL')):
                    ui.icon('settings')
                    ui.label('Control')

                with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/ABOUT')):
                    ui.icon('info')
                    ui.label('About')
            #add some spacer to content doesn't go under the fixed bar
            ui.element('div').classes('spacer')

            #When the user clicks the start simulation button run the following

            global time_crossing, sat_for_sim
            #back button
            ui.button('Back', on_click = ui.navigate.back)
            # we will navigate to here whenever a row is clicked of a specific satellite

            dt_crossing = datetime.strptime(time_crossing, '%Y-%m-%d %H:%M:%S').replace(tzinfo = timezone.utc) # convert the string to a readable time object 
        
            time_crossing_skyfield = ts.from_datetime(dt_crossing)
            
            #the object sat_for_sim: GCRS coords
            geocentric = sat_for_sim.at(time_crossing_skyfield)

            sat_r = geocentric.position.m
            sat_v = geocentric.velocity.m_per_s

            #these are for debug
            # ui.label(f'Satellite Position Vector: {sat_r}').style('font-size: 1.5em; font-weight: bold;')
            # ui.label(f'Satellite Velocity Vector: {sat_v}').style('font-size: 1.5em; font-weight: bold;')
            # ui.label("Doppler Shift Equation used from Randy L. Haupt's\nWireless Communications Systems: An Introduction").style('font-size: 1.5em; font-weight: bold; max-width: 440x; white-space: pre-line; word-break: break-word;')
            # ui.image('../Phys_Sim/media/doppler_eqn.png').style('width: 20%')
            
            #get the position and velocity of the ground stations in inertial reference frame 
            tx_geocentric = tx_pos.at(time_crossing_skyfield)
            tx_r = tx_geocentric.position.m
            tx_v = tx_geocentric.velocity.m_per_s

            rx_geocentric = rx_pos.at(time_crossing_skyfield)
            rx_r = rx_geocentric.position.m
            rx_v = rx_geocentric.velocity.m_per_s

            #debug prints
            print(f'tx position: {tx_r}')
            print(f'rx position: {rx_r}')
            print(f'tx velocity: {tx_v}')
            print(f'rx velocity: {rx_v}')

            
            #unit vector from transmitter to satellite
            k_ts = (sat_r - tx_r) / np.linalg.norm(sat_r - tx_r)
            #unit vector from satellite to receiver
            k_sr = (rx_r - sat_r) / np.linalg.norm(rx_r - sat_r)

            #user defined simulation parameters
            message = ui.input(label='Enter Message', placeholder='hello world').props('outlined dense rounded').style(
                'width: 20%; margin-bottom: 1em; font-size: 1.1em;'
            )
            ui.label('Desired transmit frequency (902 to 918 MHz)').style(
                'margin-bottom: 1em; font-size: 1.1em; font-weight: bold;'
            )
            with ui.row().style('align-items: center; width: 100%; gap: 0.5em;'):
                desired_transmit_freq = ui.slider(min=902, max=918, step=1)\
                    .props('label-always')\
                    .style('width: 10%')

                freq_input = ui.number(
                    min=902, max=918, step=1, format='%d'
                ).props('hide-spin-buttons outlined dense').style(
                    'width: 6em; font-size: 0.9em;'
                )

                # Bind values
                desired_transmit_freq.bind_value(freq_input)
                freq_input.bind_value(desired_transmit_freq)

            
            ui.label('Noise Floor (-120 to -60dBm)').style(
                'margin-bottom: 1em; font-size: 1.1em; font-weight: bold;'
            )
            with ui.row().style('align-items: center; width: 100%; gap: 0.5em;'):
                noise_power = ui.slider(min= -120, max = -60, step = 1).props('label-always').style('width: 10%')
                noise_input = ui.number(min= -120, max = -60, step = 1, format='%d').props('hide-spin-buttons outlined dense').style('width: 6em; font-size: 0.9em')
            
            ui.label('Desired SNR set to 20(dB)').style(
                'margin-bottom: 1em; font-size: 1.1em; font-weight: bold;'
            )
            # Bind values
            noise_power.bind_value(noise_input)
            noise_input.bind_value(noise_power)
            
            doppler_container =  ui.column() # store labels in this doppler container so we can easily clear them when the start_simulation button is pressed again
            
            #need to debug from here down:

            #define the loading bar (initially hidden)
            progress_bar = ui.linear_progress(show_value=False).props('indeterminate color="primary"').style('margin-top:10px; order: 2;').classes('w-64')
            #hide right away
            progress_bar.visible = False
            async def start_simulation():
                '''This function runs when the start simulation button is clicked 
                and runs handlers to produce plots for each page as necessary'''
                global txFreq, required_rep_power, bits
                progress_bar.visible = True # show the progress bar
                await asyncio.sleep(0.1) # give the UI time to update
                doppler_container.clear()
                # pass all the arguments from inputs
                mes = message.value
                if message.value == None or message.value == '':
                    mes = 'hello world' #default value
                    ui.notify('defaulting message to \'hello world\'')
                # print(mes)
                # note that the transmit freq will be in MHz
                txFreq = desired_transmit_freq.value
                if txFreq is not None:
                    txFreq = desired_transmit_freq.value*1e6 # input in MHz to Hz
                else:  
                    txFreq = 905e6 #default value is 905MHz
                    ui.notify('defaulting transmit freq to 905 MHz')
                noise = noise_power.value
                if noise is not None:
                    #convert to linear
                    noise = 10 ** ( noise / 10 ) # units will be in mW because dBm definition
                else:
                    #set default noise to -120dB
                    noise = 10 ** (-120 / 10) # units will be in mW because dBm definition
                    ui.notify('defaulting noise power to -120 dBm')

                snr = 10 ** (20/10) # default desired snr to 20 dB 
                
                #From noise: we will compute the required transmit power for transmitter and repeater so that the SNR at the receiver is at least 20dB

                #uplink doppler:            
                c = 2.99792458e8
                lambda_up = c / txFreq
                
                with doppler_container:
                    label_style = 'font-size: 1.2em; font-weight: bold; margin-bottom: 0.5em; color: #2d3748;'
                    #equation used ((kc - (vr dot khat)/lambda)/(kc - (vt dot khat)/lambda))fc kc = 1/lambda* speed of light
                    f_doppler_shifted = ((txFreq - np.dot(sat_v, k_ts) / lambda_up) / (txFreq - np.dot(tx_v, k_ts) / lambda_up)) * txFreq
                    f_delta_up = f_doppler_shifted - txFreq

                    ui.label(f'The doppler shifted frequency for uplink: {f_doppler_shifted:.7f} Hz').style(label_style)
                    ui.label(f'Delta f: {f_delta_up:.7f} Hz').style(label_style)

                    
                    #downlink doppler 
                    ui.label("The repeater simply upconverts by 10MHz and retransmits").style(label_style)
                    f_c_down = f_doppler_shifted + 10e6 # the transponder will repeat 10 MHz higher frequency everything it receives
                    
                    lambda_down = c/f_c_down
                    f_doppler_shifted = ((f_c_down - np.dot(rx_v, k_sr)/lambda_down)/(f_c_down - np.dot(sat_v, k_sr)/lambda_down)) * f_c_down
                    f_delta_down = f_doppler_shifted - f_c_down
                    ui.label(f'The doppler shifted frequency for downlink: {f_doppler_shifted:.7f} Hz').style(label_style)
                    ui.label(f'Delta f: {f_delta_down:.7f} Hz').style(label_style)

                    #time delay up and down:

                    # print(f'distance uplink debug:  {np.linalg.norm(sat_r-tx_r)/1000} km')
                    time_delay_up = np.linalg.norm(sat_r-tx_r) / c # m/m/s = s
                    ui.label(f'Time delay up: {time_delay_up:.7f} s').style(label_style)

                    time_delay_down = np.linalg.norm(rx_r - sat_r) / c 
                    ui.label(f'Time delay down: {time_delay_down:.7f} s').style(label_style)

                    #channel model:
                    #antenna gain
                    gain_tx = 10**(15/10) # 15 dB # Yagi or helical
                    gain_rx = 10**(15/10) # 15 dB # Yagi or helical
                    gain_sat = 10**(10/10) # 10 dB # helical
                    
                    #attenuation friis calculation
                    alpha_up = gain_tx * gain_sat * (lambda_up / (4 * np.pi * np.linalg.norm(sat_r - tx_r))) ** 2 # path loss attenuation
                    
                    #pick theta uniformly at random from 0 to 360 degrees 
                    THETA = np.random.uniform(0, 2*np.pi)

                    '''notes about alpha:
                        since alpha is an attenuation in power e.g. Pr/Pt = alpha
                        for amplitude we would square root alpha since in general
                        p = |x(t)|^2
                    '''
                    h_up = np.sqrt(alpha_up) * np.exp(1j * THETA) # single tap block channel model
                    
                    print(f'alpha down: {alpha_up}')
                    print(f'h_up: {h_up}')

                    #attenuation do to Friis
                    alpha_down = gain_rx * gain_sat *(lambda_down/(4*np.pi*np.linalg.norm(rx_r-sat_r)))**2 # path loss attenuation
                    print(f'alpha down: {alpha_down}')
                    #pick theta uniformly at random from 0 to 180 degrees
                    THETA = np.random.uniform(0, np.pi)
                    h_down = np.sqrt(alpha_down) * np.exp(1j * THETA) # single tap block channel model
                    print(f'hdown: {h_down}')
                    
                    #since Pr/Pt = alpha and Pr/noise = snr then Pt = N*SNR/alpha
                    noise = noise / 1000 # get noise into watts
                    #noise is going to be small 
                    required_tx_power = (snr * noise) / alpha_up # power in watts
                    required_rep_power = (snr * noise) / alpha_down #power in watts

                    ui.label(f'Required power (transmitter -> repeater) to satisfy desired SNR: {required_tx_power:.2f} W').style(label_style)
                    ui.label(f'Required power (repeater -> receiver) to satisfy desired SNR: {required_rep_power:.2f} W').style(label_style)


                #TODO simply run all of the handlers here that produce desired graphs to be used in each individual page
                #decide the amplitude of the signal so that by the time it gets to the repeater it's very
                # Calculate amplitude scaling so that the QPSK signal has required_tx_power at the repeater
                # QPSK average power is proportional to amp^2 (assuming unit average symbol energy)
                # We'll set amp so that after channel attenuation, received power = required_tx_power
                # So: amp^2 * alpha_up = required_tx_power  =>  amp = sqrt(required_tx_power / alpha_up)
                
                tx_amp = np.sqrt(required_tx_power) #we want the amplitude
                sig_gen = SigGen.SigGen(txFreq, amp = tx_amp)
                global bits
                bits = sig_gen.message_to_bits(mes) #note that this will add prefix and postfix to the bits associated wtih the message

                t, qpsk_signal = sig_gen.generate_qpsk(bits)

                sig_gen.handler(t) # run the sig gen handler

                #check if we scaled the power of the signal correctly
                print(f'does: {np.mean(np.abs(qpsk_signal) ** 2)} = {required_tx_power}')

                #define channel up
                channel_up = Channel.Channel(qpsk_signal, h_up, noise, f_delta_up, up = True)
                #apply the channel: 
                new_t, qpsk_signal_after_channel = channel_up.apply_channel(t, time_delay_up)
                #run the channel_up_handler:
                channel_up.handler(t, new_t, txFreq, SAMPLE_RATE / SYMB_RATE) #generate all the plots we want to display
                
                #amplify and upconvert:
                #we want the outgoing power to reach the required power
                Pcurr = np.mean(np.abs(qpsk_signal_after_channel) ** 2)
                gain = np.sqrt(required_rep_power / Pcurr) #this gain is used to get the power of the signal to desired power
                
                repeated_qpsk_signal = gain * np.exp(1j*2 * np.pi * 10e6 * new_t) * qpsk_signal_after_channel 

                #now we want to see if we actually got to the desired power
                #debug:
                print(f'does: {np.mean(np.abs(repeated_qpsk_signal)**2)} = {required_rep_power}')

                # run the signal through channel down
                channel_down = Channel.Channel(repeated_qpsk_signal, h_down, noise, f_delta_down, up = False)
                
                # This signal is what gets fed into the reciever
                new_t2,repeated_signal_after_channel = channel_down.apply_channel(new_t, time_delay_down)

                channel_down.handler(new_t, new_t2, txFreq + 10e6, SAMPLE_RATE / SYMB_RATE) #tune to tx + 10 MHz
                
                ###-----------------------------------
                ##TODO add channel correction here 
                #Very first step tune to what we THINK is baseband
                tuned_signal = repeated_signal_after_channel * np.exp(-1j * 2 * np.pi * (txFreq + 10e6) * new_t2)
                # run the handler for channel_correction
                global recovered_message 
                recovered_message = channel_handler(tuned_signal) # this will generate the plots we want to display on the receiver page
                #### -------------------------------------
                # channel_down = Channel.Channel()
                ui.notify('Simulation Ready')
                #hide the loading bar
                progress_bar.visible = False
                return
            
            ui.button('Start Simulation', on_click = start_simulation)
            
            # Transmitter image (clickable)
            with ui.link(target='/transmitter', new_tab = True).style('width: 10vw; position: fixed; bottom: 2vh; left: 36vw; z-index: 1000; cursor: pointer;'):
                ui.image('../Phys_Sim/media/antenna_graphic.png').style('width: 100%;')
            ui.label('Transmitter').style('position: fixed; bottom: 12vh; left: 38vw; z-index: 1005; font-weight: bold; background: rgba(255,255,255,0.7); padding: 2px 8px; border-radius: 6px;')
            ui.label('15dBi').style('position: fixed; bottom: 12vh; left: 45vw; z-index: 1008; font-weight: bold; background: rgba(255,255,255,0.7); padding: 2px 8px; border-radius: 6px;')
            # Channel 1 cloud (clickable)
            with ui.link(target='/channel1', new_tab = True).style('width: 10vw; position: fixed; bottom: 42vh; left: 43vw; z-index: 1001; cursor: pointer;'):
                ui.image('../Phys_Sim/media/cloud.png').style('width: 100%;')

            # Repeater satellite (clickable)
            with ui.link(target='/repeater', new_tab = True).style('width: 12vw; position: fixed; bottom: 64vh; left: 54vw; z-index: 1002; cursor: pointer;'):
                ui.image('../Phys_Sim/media/sattelite.png').style('width: 100%;')
            ui.label('Repeater').style('position: fixed; bottom: 76vh; left: 57vw; z-index: 1006; font-weight: bold; background: rgba(255,255,255,0.7); padding: 2px 8px; border-radius: 6px;')
            ui.label('10dBi').style('position: fixed; bottom: 76vh; left: 65vw; z-index: 1009; font-weight: bold; background: rgba(255,255,255,0.7); padding: 2px 8px; border-radius: 6px;')
            # Channel 2 cloud (clickable)
            with ui.link(target='/channel2', new_tab = True).style('width: 10vw; position: fixed; bottom: 42vh; left: 67vw; z-index: 1003; cursor: pointer;'):
                ui.image('../Phys_Sim/media/cloud.png').style('width: 100%;')

            # Receiver image (clickable)
            with ui.link(target='/receiver', new_tab=True).style('width: 10vw; position: fixed; bottom: 2vh; left: 80vw; z-index: 1004; cursor: pointer;'):
                ui.image('../Phys_Sim/media/antenna_graphic_flipped.png').style('width: 100%;')
            ui.label('Receiver').style('position: fixed; bottom: 12vh; left: 84vw; z-index: 1007; font-weight: bold; background: rgba(255,255,255,0.7); padding: 2px 8px; border-radius: 6px;')        
            ui.label('15dBi').style('position: fixed; bottom: 12vh; left: 90vw; z-index: 1008; font-weight: bold; background: rgba(255,255,255,0.7); padding: 2px 8px; border-radius: 6px;')

            # Placeholder pages for each simulation step
            @ui.page('/transmitter')
            def transmitter_page():
                ui.add_head_html('''<script>document.title = 'Transmitter';</script>''')

                # ui.button('Back', on_click=ui.navigate.back)
                ui.label('Transmitter Page').style('font-size: 3em; font-weight: bold; text-align: center; display: block; width: 100%;')
                with ui.element('div').classes('flex-container'):
                    #bit sequence with prefix/postifx labeled
                    #TODO

                    #show upsampled bits sub plot one on top of the other real and imaginary
                    zoomable_image('media/tx_upsampled_bits.png')
                    # ui.label('Notice that energy is very spread out in the spectrum because impulses in time are infinite in frequency').style('font-size: 1.5em; font-weight: bold;')
                    zoomable_image('media/tx_upsampled_bits_fft.png')
                    # ui.label('These upsampled bits are pulse shaped with the following filter:').style('font-size: 1.5em; font-weight: bold;')
                    zoomable_image('media/tx_rrc.png')
                    
                    #show the pulse shaping Re/Im
                    zoomable_image('media/tx_pulse_shaped_bits.png')
                    
                    #show the baseband FFT
                    zoomable_image('media/tx_pulse_shaped_fft.png')
                    
                    #show the constellation plot
                    zoomable_image('media/tx_constellation.png')
            
            @ui.page('/channel1')
            def channel1_page():
                ui.add_head_html('''<script>document.title = 'Channel Up';</script>''')

                # ui.button('Back', on_click=ui.navigate.back)
                ui.label('Channel Uplink Page').style('font-size: 3em; font-weight: bold; text-align: center; display: block; width: 100%;')
                with ui.element('div').classes('flex-container'):
                    
                    #show information about h
                    zoomable_image('media/channel_up_h_phase.png')

                    # constellation plot of incoming signal
                    zoomable_image('media/channel_up_incoming_tuned_constellation.png')

                    # constellation plot of outgoing signal
                    zoomable_image('media/channel_up_outgoing_tuned_constellation.png')

            @ui.page('/repeater')
            def repeater_page():     
                ui.add_head_html('''<script>document.title = 'Repeater';</script>''')
                ui.label('Repeater Page').style('font-size: 3em; font-weight: bold; text-align: center; display: block; width: 100%;')
                ui.label(f'The repeater will retransmit at {required_rep_power} W').style(
                    'font-size: 1.5em; font-weight: bold; margin-top: 1em; text-align: center; display: block; width: 100%;'
                )
                ui.label(f'The repeater takes in the signal and sends it back out, upconverted 10 MHz').style(
                    'font-size: 1.5em; font-weight: bold; margin-top: 1em; text-align: center; display: block; width: 100%;'
                )

            @ui.page('/channel2')
            def channel2_page():
                ui.add_head_html('''<script>document.title = 'Channel Down';</script>''')

                # ui.button('Back', on_click=ui.navigate.back)
                ui.label('Channel Downlink Page').style('font-size: 3em; font-weight: bold; text-align: center; display: block; width: 100%;')
                with ui.element('div').classes('flex-container'):                
                    #show information about h
                    zoomable_image('media/channel_down_h_phase.png')
                    # constellation plot of incoming signal
                    zoomable_image('media/channel_down_incoming_tuned_constellation.png')
                    # constellation plot of outgoing signal
                    zoomable_image('media/channel_down_outgoing_tuned_constellation.png')


            @ui.page('/receiver')
            def receiver_page():
                ui.add_head_html('''<script>document.title = 'Receiver';</script>''')

                # ui.button('Back', on_click=ui.navigate.back)
                ui.label('Receiver Page').style('font-size: 3em; font-weight: bold; text-align: center; display: block; width: 100%;')
                with ui.element('div').classes('flex-container'):
                    
                    # constellation plot of the incoming signal and fft
                    zoomable_image('media/rx_incoming.png')

                    # constellation plot of the incoming signal and fft after corse frequency correction
                    zoomable_image('media/coarse_correction.png')

                    #binary search CAF convergence
                    zoomable_image('media/binary_search_convergence.png')

                    # show IQ before phase correction
                    zoomable_image('media/pre_phase_correction_constellation.png')
                
                    # show phase correction
                    zoomable_image('media/phase_offset.png')
                
                    # show IQ after phase correction
                    zoomable_image('media/phase_corrected_constellation.png')

                    # show start and end correlation and the indecies to start and end of the message
                    zoomable_image('media/start_end_correlation.png')
                    # show fine frequency correction constellation and fft

                    zoomable_image('media/fine_correction.png')

                    # show the final constellation plot after all corrections
                    zoomable_image('media/clean_signal.png')

                # compute throughput
                global bits
                info_bit_ratio = (len(bits) - len(START_MARKER) - len(END_MARKER)) / len(bits)
                throughput = 2 * SYMB_RATE * info_bit_ratio # 2 bits per symbol * symbols_per_second = bits/s * info_bit_ratio = useful_bits/s
                #show the throughput image
                
                #
                ui.label(f'Calculated throughput: {throughput} bps').style('font-size: 1.5em; font-weight: bold; margin-top: 1em;')
                #show the final recovered message 
                ui.label(f'Recovered Message: {recovered_message}').style('font-size: 1.5em; font-weight: bold; margin-top: 1em;')


# Control Page----------------------------------------------------------------------------------------------------------------------
@ui.page('/CONTROL')
def control_page():
    inject_head_style('Control')

    with ui.element('div').classes('glass-bar'):
        with ui.element('div').classes('item').on('click', lambda: (ui.navigate.to('/'))):
            ui.icon('home')
            ui.label('Home')
        with ui.element('div').classes('item').on('click', lambda: (ui.navigate.to('/SIMULATE'))):
            ui.icon('code')
            ui.label('Simulation')

        with ui.element('div').classes('item').on('click', lambda: (ui.navigate.to('/CONTROL'))):
            ui.icon('settings')
            ui.label('Control')

        with ui.element('div').classes('item').on('click', lambda: (ui.navigate.to('/ABOUT'))):
            ui.icon('info')
            ui.label('About')
    #add some spacer to content doesn't go under the fixed bar
    ui.element('div').classes('spacer')

    with ui.row().style('justify-content: center; align-items: flex-end; width: 100%; gap: 5vw; margin-top: 4vh;'):
        # Use a variable for min-height to ensure both columns match
        phone_min_height = "520px"
        phone_width = "340px"
        phone_style = f'background: #f5f5f7; border-radius: 32px; box-shadow: 0 4px 24px #bbb; width: {phone_width}; min-height: {phone_min_height}; height: {phone_min_height}; padding: 24px 16px 16px 16px; position: relative; display: flex; flex-direction: column; justify-content: flex-end;'

        # Left phone (Sender)
        with ui.column().style(phone_style):
            with ui.column().style('align-items: center; width: 100%;'):
                # Small circular "contact photo"
                ui.image('media/TX_contact.png').style('width: 40px; height: 40px; border-radius: 50%; object-fit: cover; margin-bottom: 0.01em; box-shadow: 0 4px 12px rgba(0,0,0,0.25);')
                ui.label('Transmitter').style(
                    'font-size: 1.1em; margin-bottom: 1em; text-align: center; font-family: "SF Pro", "SF Pro Display", "San Francisco", "Segoe UI", "Arial", sans-serif;'
                )
            # Message bubbles area (could be enhanced to show message history)
            with ui.row().style('justify-content: flex-end; width: 100%; min-height: 240px; flex: 1;'):
                message_bubble = ui.label('No messages yet...').style(
                    'background: #007AFF; color: white; border-radius: 32px; padding: 8px 16px; margin: 4px 0; font-size: 1.1em; max-width: 70%; text-align: right; font-family: "SF Pro", "SF Pro Display", "San Francisco", "Segoe UI", "Arial", sans-serif;'
                )
                message_bubble.visible = False  # Hide initially
            # Input row
            with ui.row().style('width: 100%; align-items: center; margin-top: 2em;'):
                text_input = ui.input(placeholder="Type a message...").props('rounded outlined dense').style('flex: 1; font-size: 1.1em;')
                async def handle_send():
                    msg = text_input.value
                    if not msg:
                        ui.notify('Please enter a message to send.')
                        return
                    message_bubble.text = msg  # Update message bubble
                    message_bubble.visible = True  # Show message bubble
                    text_input.value = ""  # Clear input
                    await send_message(msg)

                ui.button(on_click=handle_send)\
                    .props('color=primary round dense icon="arrow_upward"').style('margin-left: 0.3em; width: 20px; height: 20px; font-size: 1.3em; background-color: #1976d2; color: white;')\
                    .classes('q-btn--fab')

        # Right phone (Receiver)
        with ui.column().style(phone_style):
            with ui.column().style('align-items: center; width: 100%;'):
                # Small circular "contact photo"
                ui.image('media/RX_contact.png').style('width: 40px; height: 40px; border-radius: 50%; object-fit: cover; margin-bottom: 0.01em; box-shadow: 0 4px 12px rgba(0,0,0,0.25);')
                ui.label('Receiver').style('font-size: 1.1em; margin-bottom: 1em; text-align: center; font-family: "SF Pro", "SF Pro Display", "San Francisco", "Segoe UI", "Arial", sans-serif;')
            # Message bubbles area (could be enhanced to show message history)
            with ui.row().style('justify-content: flex-start; width: 100%; min-height: 340px; flex: 1;'):
                # 3-dot loading indicator
                received_bubble = ui.label('Waiting for messages...').style(
                    'background: #747474; color: white; border-radius: 32px; padding: 8px 16px; margin: 4px 0; font-size: 1.1em; max-width: 70%; text-align: left; font-family: "SF Pro", "SF Pro Display", "San Francisco", "Segoe UI", "Arial", sans-serif;'
                )
                received_bubble.visible = False  # Hide initially
                # Create a "bubble" for loading dots, styled like the received_bubble
                loading_dots = ui.html(
                    '''
                    <div style="background: #747474; color: white; border-radius: 32px; padding: 8px 16px; margin: 4px 0; font-size: 1.1em; max-width: 70%; text-align: left; display: flex; align-items: center; min-height: 24px; height: 40px; justify-content: center;">
                        <span class="dot1" style="display: inline-block; width: 8px; height: 8px; aspect-ratio: 1 / 1; margin: 0 3px; background: #bbb; border-radius: 50%; animation: bounce1 1.2s infinite alternate;"></span>
                        <span class="dot2" style="display: inline-block; width: 8px; height: 8px; aspect-ratio: 1 / 1; margin: 0 3px; background: #888; border-radius: 50%; animation: bounce2 1.2s 0.2s infinite alternate;"></span>
                        <span class="dot3" style="display: inline-block; width: 8px; height: 8px; aspect-ratio: 1 / 1; margin: 0 3px; background: #444; border-radius: 50%; animation: bounce3 1.2s 0.4s infinite alternate;"></span>
                    </div>
                    <style>
                    @keyframes bounce1 {
                        0% { transform: translateY(0); background: #bbb;}
                        50% { background: #eee;}
                        100% { transform: translateY(-2px); background: #bbb;}
                    }
                    @keyframes bounce2 {
                        0% { transform: translateY(0); background: #888;}
                        50% { background: #ccc;}
                        100% { transform: translateY(-2px); background: #888;}
                    }
                    @keyframes bounce3 {
                        0% { transform: translateY(0); background: #444;}
                        50% { background: #999;}
                        100% { transform: translateY(-2px); background: #444;}
                    }
                    </style>
                    '''
                )
                loading_dots.visible = False


    async def send_message(message):
        """Sends the message to the hardware and waits for a response."""
        received_bubble.visible = False #start the receive bubble as hidden
        if not message:
            ui.notify('Please enter a message to send.')
            return
        loading_dots.visible = True
        await asyncio.sleep(0.1)  # give the UI time to update

        #tell the transmitter and repeater to transmit but don't wait on them to finish up
        try: 
            ssh_host = 'empire@empire'
            command = f'cd /home/empire/Documents/InternProj2025/Final_Product/transmitter && nohup ./transmit.bash "{message}" > output.log 2>&1 &'
            async with asyncssh.connect('empire', username = 'empire', password='password', known_hosts=None) as conn:
                await conn.create_process(command)
        except (OSError, asyncssh.Error) as e:
            ui.notify(f'SSH error: {e}')
            loading_dots.visible = False
            return
        
        
        # ZMQ request to the receiver (same LAN IP or host name)
        try: 
            # print('did we get here')
            context = zmq.asyncio.Context()
            Rx = context.socket(zmq.REQ)
            Rx.connect('tcp://10.232.62.2:5555') # Receiver IP + port 5555
            await Rx.send_string("SEND")
            #we'll asynchronously listen for a reply from the receiver computer
            decoded_message = await Rx.recv_string()
        except:
            ui.notify(f'ZMQ error')
            loading_dots.visible = False
            return
        
        loading_dots.visible = False
        
        # put the message here
        received_bubble.text = f'{decoded_message}'  # Update received bubble
        received_bubble.visible = True

        #ui.notify('Message sent!')
        return
            



@ui.page('/ABOUT')
def about_page():
    
    # Serve the media folder statically so images can be accessed via /static/media/...
    media_dir = pathlib.Path(__file__).parent / "media"
    app.add_static_files('/static/media', str(media_dir))
    ui.add_head_html('''
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
            <script>
            document.title = 'About';
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" integrity="sha512-yvLqEXxCzCyxUeHkMgPh3/jtdMELH7BykTbk+8vwFpD2Z6jszD0Q5YQ3fvLRvAVNsmF29eEobTVE+q6d+pc+xg==" crossorigin="anonymous" referrerpolicy="no-referrer" />
            </script>         
            <style>
            .glass-bar {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                min-width: 100vw;
                height: 60px;
                display: flex;
                gap: 20px;
                align-items: center;
                justify-content: center;
                backdrop-filter: blur(10px);
                background: rgba(0, 0, 0, 0.4);
                -webkit-backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                z-index: 1000;
                padding: 0 30px;
                box-sizing: border-box;
                color: white;
            }

            .glass-bar .item {
                display: flex;
                align-items: center;
                cursor: pointer;
                gap: 8px;
                font-size: 18px;
                color: white;
                transition: background 0.2s;
            }

            .glass-bar .item:hover {
                scale: 1.05;
            }

            .spacer {
                height: 60px; /* reserve space under the fixed bar */
            }     

            * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            }

            img {
            max-width: 100%;
            }

            body {
            font-family: system-ui, sans-serif;
            font-size: 16px;
            line-height: 1.5;
            color: #0f0f0f;
            background-color: #ecedef;
            padding: 50px;
            }

            h1 {
            text-align: center;
            margin-bottom: 4rem;
            }

            .users {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-flow: wrap;
            gap: 30px;
            }

            .user {
            position: relative;
            z-index: 1;
            width: 250px;
            /* height: 350px; */
            /* 350/250 = 1.4 */
            aspect-ratio: 1 / 1.4;
            padding: 1rem;
            border-radius: 20px;
            background-color: #fff;
            box-shadow: 0 30px 30px 5px #d6d9e2;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            gap: 1rem;
            }

            .user-img-wrap {
            position: relative;
            width: 117px;
            aspect-ratio: 1;
            padding: 7px;
            border-radius: 100%;
            }

            .user-img-wrap::after {
            position: absolute;
            z-index: -1;
            content: "";
            inset: 0;
            border-radius: 100%;

            background: linear-gradient(
                #4cd964,
                #5ac8fa,
                #007aff,
                #7dc8e8,
                #5856d6,
                #ff2d55
            );

            opacity: 0;
            transition: opacity 1s;

            animation: rotate 4s linear infinite;
            animation-play-state: paused;
            filter: saturate(2) blur(10px);
            }

            .user:hover .user-img-wrap::after {
            opacity: 1;
            animation-play-state: running;
            }

            @keyframes rotate {
            to {
                rotate: 360deg;
            }
            }

            .user-img {
            aspect-ratio: 1;
            border-radius: 100%;
            overflow: hidden;
            }

            .user-meta {
            text-align: center;
            }

            .user-name {
            font-size: 20px;
            font-size: 1.25rem;
            font-weight: 500;
            }

            .user-role {
            font-size: 14px;
            font-size: 0.875rem;
            color: #a0a2b6;
            margin-bottom: 1rem;
            }
                     
            .user-school {
            font-size: 14px;
            font-size: 0.875rem;
            color: #a0a2b6;
            margin-bottom: 1rem;          
            }

            .user-profiles {
            font-size: 1rem;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            }
            </style>          
            ''')
    with ui.element('div').classes('glass-bar'):
        with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/')):
            ui.icon('home')
            ui.label('Home')
        with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/SIMULATE')):
            ui.icon('code')
            ui.label('Simulation')

        with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/CONTROL')):
            ui.icon('settings')
            ui.label('Control')

        with ui.element('div').classes('item').on('click', lambda: ui.navigate.to('/ABOUT')):
            ui.icon('info')
            ui.label('About')
    #add some spacer to content doesn't go under the fixed bar
    ui.element('div').classes('spacer')
    with ui.row().style('width: 80%; justify-content: space-between; align-items: center; margin: 0 auto;'):
        with ui.column().style('width: 40%; min-width: 200px;'):
            #left image
            ui.image('https://www.rincon.com/assets/imgs/slides/satellite-1.jpg')
        with ui.column().style('width: 55%; min-width: 200px;'):
            #right text
            # Add a project title at the top of the About page
            ui.label("Red Mountain Internship Project").style(
                'font-size: 2.5em; font-weight: bold; margin-bottom: 0.5em; text-align: center; display: block; width: 100%;'
            )
            ui.label("In Rincon Research Coorporation's Internship program in the summer of 2025 four bright interns from a variety of diciplines came together to simulate and realize a bent pipe communication system. Bent pipe communication systems are often used in scenarios where line of sight between a transmitter and a receiver is obscured, or in the scenario where certain frequency bands are denied by an interferer. The task involved extensive research and experimentation with unfamiliar digital signal processing concepts including QPSK modulation, pulse shaping, finite impulse response filters, sampling theory, the cross ambiguity function (CAF), and detection. The experience was rewarding and the final product is nothing shy of cool.").style(
                'font-size: 1.3em; margin-bottom: 2em; text-align: center; display: block; width: 100%;'
            )
    #add some spacer
    ui.element('div').classes('spacer')

    # Simulation
    with ui.row().style('width: 80%; justify-content: center; align-items: flex-start; margin: 0 auto;'):
        with ui.column().style('width: 50%; align-items: flex-start;'):
            ui.label("Simulation Overview").style(
                'font-size: 2.5em; font-weight: bold; text-align: center; width: 100%;'
            )
            ui.label("Explore the digital simulation of a bent pipe satellite communication system.").style(
                'font-size: 1.3em; text-align: center; width: 100%;'
            )
            ui.label(
                "The simulation retrieves up-to-date TLE (Two-Line Element) data for recently launched satellites from CelesTrak, enabling dynamic visualization of satellite orbits and line-of-sight paths between ground stations and satellites. For each selected satellite, the system computes the time and geometry of closest approach, allowing users to choose a satellite as the transponder in a bent pipe communication scenario. Leveraging real orbital data, the simulation estimates time delays and Doppler shifts to create a realistic channel model. Users can interactively step through each stage of the communication link—from transmitter to satellite transponder and onward to the receiver—gaining insight into the physical and signal processing aspects of satellite communications."
            ).style('font-size: 1.1em;')
        with ui.column().style('width: 45%; align-items: flex-start;'):
            ui.image('media/Sim_screenshot.png').style('width: 90%;')
   
    #add some spacer
    ui.element('div').classes('spacer')
    
    #Control
    with ui.row().style('width: 80%; justify-content: center; align-items: flex-start; margin: 0 auto;'):
        with ui.column().style('width: 45%; align-items: center; '):
            #wrap the image in a row so we can center it
            ui.image('media/Control_Interface.png').style('width: 90%; margin-top: 9em;').force_reload()
        with ui.column().style('width: 50%; align-items: flex-start;'):
            ui.label("Control Overview").style(
                'font-size: 2.5em; font-weight: bold; text-align: center; width: 100%;'
            )
            ui.label("Bring simulated results to the real world").style(
                'font-size: 1.3em; text-align: center; width: 100%;'
            )
            ui.label("There are three devices as there are in the simulated scenario:"
            ).style('font-size: 1.1em;')
            with ui.row().style('width: 100%; justify-content: center; align-items: flex-start; margin-top: 2em;'):
                with ui.column().style('width: 30%; align-items: center;'):
                    ui.image('media/tx_device.png').style('width: 90%;')
                    ui.label('Transmitter (VSG60A)').style('font-size: 1.2em; font-weight: bold; margin-top: 0.5em;')
                with ui.column().style('width: 30%; align-items: center;'):
                    ui.image('media/repeater_device.png').style('width: 90%;')
                    ui.label('Repeater (BladeRF)').style('font-size: 1.2em; font-weight: bold; margin-top: 0.5em;')
                with ui.column().style('width: 30%; align-items: center;'):
                    ui.image('media/rx_device.png').style('width: 60%;')
                    ui.label('Receiver (RTL-SDR)').style('font-size: 1.2em; font-weight: bold; margin-top: 0.5em;')
            ui.label('The control page allows users to input a custom message for transmission. Once submitted, the message is modulated using QPSK and passed to an asynchronous message handler, which coordinates communication between three core devices: the transmitter, repeater, and receiver. The transmitter sends the modulated signal to the repeater, which amplifies and upconverts all incoming signals to a higher frequency. The receiver then detects the presence of the signal, applies channel correction, and demodulates the signal to recover the original message.').style('font-size: 1.1em;')

    #add some spacer
    ui.element('div').classes('spacer')
    ui.label("Authors").style('font-size: 2.5em; font-weight: bold; text-align: center; width: 100%;')
    with ui.row().style('width: 100%; justify-content: center;'):
        ui.html('''       
    <div class="users">
      
        <div class="user">
        <div class="user-img-wrap">
          <div class="user-img">
        <img src="/static/media/Skylar.jpg">
          </div>
        </div>
        <div class="user-meta">
          <div class="user-name">
        Skylar Harris
          </div>
          <div class = 'user-role'>
        ZMQ, Signal Processing, Hardware Implementation
          </div>
          <div class="user-school">
        University of Colorado Boulder
          </div>
          
          <div class="user-profiles">
        <a href="https://www.linkedin.com/in/skylar-harris-82aba52a4?trk=people-guest_people_search-card" target="_blank" title="LinkedIn"><i class="fa-brands fa-linkedin"></i></a>
        <a href="https://github.com/skha3371" target="_blank" title="GitHub"><i class="fa-brands fa-github"></i></a>

            </div>
        </div>
      </div>

    <div class="user">
        <div class="user-img-wrap">
          <div class="user-img">
        <img src="/static/media/Jorge.JPG">
          </div>
        </div>
        <div class="user-meta">
          <div class="user-name">
        Jorge Hernandez
          </div>
          <div class = 'user-role'>
        Channel Correction, Software-hardware Integration
          </div>
          <div class="user-school">
        Purdue University
          </div>
          
          <div class="user-profiles">
        <a href="https://www.linkedin.com/in/jorge-hernandez-957190242" target="_blank" title="LinkedIn"><i class="fa-brands fa-linkedin"></i></a>
        <a href="https://github.com/jorgeh309" target="_blank" title="GitHub"><i class="fa-brands fa-github"></i></a>

            </div>
        </div>
      </div>
                
    <div class="user">
        <div class="user-img-wrap">
          <div class="user-img">
        <img src="/static/media/Kobe.JPG">
          </div>
        </div>
        <div class="user-meta">
          <div class="user-name">
        Kobe Prior
          </div>
          <div class = 'user-role'>
        GUI, Satellite Simulation, Channel Model
          </div>
          <div class="user-school">
        Colorado School of Mines
          </div>
          
          <div class="user-profiles">
        <a href="https://www.youtube.com/@kobetutors/featured" target="_blank" title="YouTube"><i class="fa-brands fa-youtube"></i></a>
        <a href="https://www.linkedin.com/in/kobeprior" target="_blank" title="LinkedIn"><i class="fa-brands fa-linkedin"></i></a>
        <a href="https://github.com/kobeprior99" target="_blank" title="GitHub"><i class="fa-brands fa-github"></i></a>
          </div>
        </div>
      </div>
                

    <div class="user">
        <div class="user-img-wrap">
          <div class="user-img">
        <img src="/static/media/Trevor_cropped.jpeg">
          </div>
        </div>
        <div class="user-meta">
          <div class="user-name">
        Trevor Wiseman
          </div>
          <div class = 'user-role'>
        Signal Processing, Detection, Hardware Implementation
          </div>
          <div class="user-school">
        Brigham Young University
          </div>
          
          <div class="user-profiles">
        <a href="https://www.linkedin.com/in/trevor-w-7a5889ba" target="_blank" title="LinkedIn"><i class="fa-brands fa-linkedin"></i></a>
        <a href="https://github.com/kobeprior99" target="_blank" title="GitHub"><i class="fa-brands fa-github"></i></a>
          </div>
        </div>
      </div>

    </div>
        ''')


#run the GUI
ui.run()
