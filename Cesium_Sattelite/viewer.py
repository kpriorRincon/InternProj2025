# Author: Kobe Prior
# Date: June 10
# Purpose: This file provides a NiceGUI-based web interface for selecting recently launched satellites,
#          generating their TLE-based CZML data, and visualizing them in a Cesium viewer.

#a comment to test push from vm 
# import necessary libraries
from nicegui import ui, app
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
# function I made to get current TLE data
from get_TLE import get_up_to_date_TLE
# importing classes
import Sig_Gen as SigGen
import Channel as Channel
from config import *
from binary_search_caf import channel_handler

from satellite_czml import satellite_czml
'''note ctrl click satellite_czml then comment out satellites = {} because it isn't instance specific then
at the beginning of __init__() add self.satellites = {}
at the top of the class from datetime import datetime, timedelta, timezone
also replace all instances of datetime.utcnow() with datetime.now(timezone.utc)'''
#downgrade pygeoif: pip install pygeoif==0.7.0

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
recovered_message = None # this will be set in the simulation page when we recover the message

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


ui.number(label='How many Satellites?', min=1, max=10, step=1,
          on_change=update_text_boxes).style('width: 10%')
ui.button('Submit', on_click=submit, color='positive').style('order: 3;')


@ui.page('/Cesium_page')
def Cesium_page():
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
        
        # get this files working directory
        html_directory = os.path.dirname(__file__) #get the directory you're in
        # add the files available
        app.add_static_files('/static', html_directory)
    #cesium page take up the right 70 percent of the page
    ui.html(
        f'''
        <div style="position: fixed; top: 0; right: 0; width: 70vw; height: 95vh; border: none; margin: 1vh 1vw 0 0; padding: 0; overflow: hidden; z-index: 999999; box-shadow: 0 0 10px rgba(0,0,0,0.2); background: #fff; border-radius: 12px;">
            <iframe style="width: 100%; height: 100%; border: none;" src="/static/viewer.html?t={int(time.time())}"></iframe>
        </div>
        '''
    )

    @ui.page('/simulation_page')
    def simulation_page(): 
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
        ui.label("Doppler Shift Equation used from Randy L. Haupt's\nWireless Communications Systems: An Introduction").style('font-size: 1.5em; font-weight: bold; max-width: 440x; white-space: pre-line; word-break: break-word;')
        ui.image('../Phys_Sim/media/doppler_eqn.png').style('width: 20%')
        
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
        message = ui.input(label='Enter Message', placeholder='hello world').style(
            'width: 10%; margin-bottom: 1em; font-size: 1.1em;'
        )
        ui.label('Desired transmit frequency (MHz)').style(
            'margin-bottom: 1em; font-size: 1.1em; font-weight: bold;'
        )
        desired_transmit_freq = ui.slider(min=902,max = 918, step=1).props('label-always').style('width: 10%')
        
        ui.label('Noise Floor (dBm)').style(
            'margin-bottom: 1em; font-size: 1.1em; font-weight: bold;'
        )
        noise_power = ui.slider(min=-120,max = -60, step=1).props('label-always').style('width: 10%')
        
        ui.label('Desired SNR set to 20(dB)').style(
            'margin-bottom: 1em; font-size: 1.1em; font-weight: bold;'
        )
        
        doppler_container =  ui.column() # store labels in this doppler container so we can easily clear them when the start_simulation button is pressed again
        
        #need to debug from here down:
        def start_simulation():
            '''This function runs when the start simulation button is clicked 
            and runs handlers to produce plots for each page as necessary'''
            doppler_container.clear()
            # pass all the arguments from inputs
            mes = message.value
            if message.value == '':
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
            sig_gen = SigGen.SigGen(txFreq, amp=tx_amp)

            bits = sig_gen.message_to_bits(mes)#note that this will add prefix and postfix to the bits associated wtih the message

            t, qpsk_signal = sig_gen.generate_qpsk(bits)

            sig_gen.handler(t) # run the sig gen handler

            #check if we scaled the power of the signal correctly
            print(f'does: {np.mean(np.abs(qpsk_signal)**2)} = {required_tx_power}')

            #define channel up
            channel_up = Channel.Channel(qpsk_signal, h_up, noise, f_delta_up, up = True)
            #apply the channel: 
            new_t, qpsk_signal_after_channel = channel_up.apply_channel(t, time_delay_up)
            #run the channel_up_handler:
            channel_up.handler(t, new_t, txFreq, SAMPLE_RATE / SYMB_RATE) #generate all the plots we want to display
            
            #amplify and upconvert:
            #we want the outgoing power to reach the required power
            Pcurr = np.mean(np.abs(qpsk_signal_after_channel)**2)
            gain = np.sqrt(required_rep_power/Pcurr)#this gain is used to get the power of the signal to desired power
            repeated_qpsk_signal = gain * np.exp(1j*2 * np.pi * 10e6 * new_t) * qpsk_signal_after_channel
            #generate a plot of the tuned fft of the qpsk_signal_after_channel for the repeater page
            repeated_qpsk_signal_tuned = repeated_qpsk_signal * np.exp(-1j * 2 * np.pi * txFreq * new_t)
            N = len(repeated_qpsk_signal_tuned)
            fft_repeated = np.fft.fftshift(np.fft.fft(repeated_qpsk_signal_tuned))
            freqs_repeated = np.fft.fftshift(np.fft.fftfreq(N, d = 1 / SAMPLE_RATE))

            plt.figure(figsize=(10, 6))
            plt.plot(freqs_repeated, 20 * np.log10(np.abs(fft_repeated)))
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Magnitude (dB)")
            plt.title("FFT of Repeated Signal (Tuned to Transmit Frequency)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('media/repeater_fft.png', dpi=300)
            plt.close()

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
            return
        
        ui.button('start simulation', on_click=start_simulation)
        
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
            # ui.button('Back', on_click=ui.navigate.back)
            ui.label('Transmitter Page').style('font-size: 2em; font-weight: bold;')
            ui.label('This is a placeholder for the transmitter simulation step.')
            
            with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
                #bit sequence with prefix/postifx labeled
                #TODO
                #show upsampled bits sub plot one on top of the other real and imaginary
                ui.image('media/tx_upsampled_bits.png').style('width: 50%').force_reload()
                ui.label('Notice that energy is very spread out in the spectrum because impulses in time are infinite in frequency').style('font-size: 1.5em; font-weight: bold;')
                ui.image('media/tx_upsampled_bits_fft.png').style('width: 50%').force_reload()
                ui.label('These upsampled bits are pulse shaped with the following filter:').style('font-size: 1.5em; font-weight: bold;')
                ui.image('media/tx_rrc.png').style('width: 30%') #don't need to force reload because it doesn't change between siulations
                #show the pulse shaping Re/Im
                ui.image('media/tx_pulse_shaped_bits.png').style('width: 50%').force_reload()
                #show the baseband FFT
                ui.image('media/tx_pulse_shaped_fft.png').style('width: 50%').force_reload()
                #show the constellation plot
                ui.image('media/tx_constellation.png').style('width: 50%').force_reload()
        @ui.page('/channel1')
        def channel1_page():
            # ui.button('Back', on_click=ui.navigate.back)
            ui.label('Channel Uplink Page').style('font-size: 2em; font-weight: bold;')
            ui.label('This is a placeholder for the first channel simulation step.')

            with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
                
                #show information about h
                ui.image('media/channel_up_h_phase.png').style('width: 40%;').force_reload()

                with ui.row().style('width:100%'):
                    # constellation plot of incoming signal
                    ui.image('media/channel_up_incoming_tuned_constellation.png').style('width: 40%').force_reload()
                    # tune to baseband and show the fft
                    ui.image('media/channel_up_incoming_tuned_fft.png').style('width: 40%; align-self: center;').force_reload()
                ui.label("Note that the tuned signal is tuned based on the transmit carrier so the frequency offset from doppler manifests as phase smearing of the symbols").style('font-size: 1.2em; font-weight: bold; white-space: normal; word-break: break-word;')
                ui.label('Also note that this interpreted symbols are not aligned with the delayed signal').style('font-size: 1.2em; font-weight: bold; white-space: normal; word-break: break-word;')
                
                with ui.row().style('width:100%'):
                    # constellation plot of outgoing signal
                    ui.image('media/channel_up_outgoing_tuned_constellation.png').style('width: 40%').force_reload()
                    # fft outgoing
                    ui.image('media/channel_up_outgoing_tuned_fft.png').style('width: 40%; align-self: center;').force_reload()

                # ui.image('media/channel_up_incoming_time.png').style('width: 50%;').force_reload()
                # ui.image('media/channel_up_outgoing_time.png').style('width: 50%;').force_reload()

        @ui.page('/repeater')
        def repeater_page():
            # ui.button('Back', on_click=ui.navigate.back)
            ui.label('Repeater Page').style('font-size: 2em; font-weight: bold;')
            ui.label('This is a placeholder for the repeater simulation step.')
            with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
                ui.image('media/repeater_fft.png').style('width: 40%;').force_reload()

        @ui.page('/channel2')
        def channel2_page():
            # ui.button('Back', on_click=ui.navigate.back)
            ui.label('Channel Downlink Page').style('font-size: 2em; font-weight: bold;')
            ui.label('This is a placeholder for the second channel simulation step.')
            with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
                
                #show information about h
                ui.image('media/channel_down_h_phase.png').style('width: 40%;').force_reload()

                with ui.row().style('width:100%'):
                    # constellation plot of incoming signal
                    ui.image('media/channel_down_incoming_tuned_constellation.png').style('width: 40%').force_reload()
                    # tune to baseband and show the fft
                    ui.image('media/channel_down_incoming_tuned_fft.png').style('width: 40%; align-self: center;').force_reload()
                ui.label("Note that the tuned signal is tuned based on the transmit carrier so the frequency offset from doppler manifests as phase smearing of the symbols").style('font-size: 1.2em; font-weight: bold; white-space: normal; word-break: break-word;')
                with ui.row().style('width:100%'):
                    # constellation plot of outgoing signal
                    ui.image('media/channel_down_outgoing_tuned_constellation.png').style('width: 40%').force_reload()
                    # fft outgoing
                    ui.image('media/channel_down_outgoing_tuned_fft.png').style('width: 40%; align-self: center;').force_reload()

        @ui.page('/receiver')
        def receiver_page():
            # ui.button('Back', on_click=ui.navigate.back)
            ui.label('Receiver Page').style('font-size: 2em; font-weight: bold;')
            ui.label('This is a placeholder for the receiver simulation step.')
            with ui.column().style('width: 100%; justify-content: center; align-items: center;'):
                
                # constellation plot of the incoming signal and fft
                with ui.row().style('width:100%'):
                       # constellation plot of outgoing signal
                    ui.image('media/channel_down_outgoing_tuned_constellation.png').style('width: 40%').force_reload()
                    # fft outgoing
                    ui.image('media/channel_down_outgoing_tuned_fft.png').style('width: 40%; align-self: center;').force_reload()
                
                # constellation plot of the incoming signal and fft after LPF
                # with ui.row().style('width: 100%; justify-content: center; align-items: center;'):
                #     ui.image('media/receiver_constellation_lpf.png').style('width: 40%').force_reload()

                # constellation plot of the incoming signal and fft after corse frequency correction
                # with ui.row().style('width: 100%; justify-content: center; align-items: center;'):
                #      ui.image('media/receiver_constellation_coarse_freq.png').style('width: 40').force_reload()

                #binary search CAF convergence
                with ui.row().style('width: 100%; justify-content: center; align-items: center;'):
                    ui.image('media/binary_search_convergence.png').style('width: 40%').force_reload()

                # show phase correction
                with ui.row().style('width: 100%; justify-content: center; align-items: center;'):
                    ui.image('media/phase_offset.png').style('width: 40%').force_reload()

                # show start and end correlation
                with ui.row().style('width: 100%; justify-content: center; align-items: center;'):
                    ui.image('media/start_correlation.png').style('width: 40%').force_reload()
                    ui.image('media/end_correlation.png').style('width: 40%').force_reload()
               
                # show fine frequency correction constellation and fft
                # with ui.row().style('width: 100%; justify-content: center; align-items: center;'):
                #     ui.image('media/receiver_constellation_fine_freq.png').style('width: 40%').force_reload()

                # show the final recovered bits 
                
                #show the final recovered message 
                ui.label(f'Recovered Message: {recovered_message}').style('font-size: 1.5em; font-weight: bold; margin-top: 1em;')
ui.run()
