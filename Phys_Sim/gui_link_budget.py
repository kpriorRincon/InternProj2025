from nicegui import ui
import numpy as np
import sim_link_budget 

#################################################
#
#   Author: Trevor Wiseman
#
#################################################

# Function to calculate and display results
def calculate_uplink(Gt_value, Pt_value, f_value, B_value, Gr_value, R_value):
    # Call the compute_parameters function
    lam, Pn, Pr, SNR_dB, H = sim_link_budget.compute_parameters(Gt_value, Pt_value, f_value, B_value, Gr_value, R_value)

    # Update the result labels with the computed values
    result_labels['Wavelength (m)'].text = f"Wavelength (m): {lam:.6f}"
    result_labels['Noise Power (W)'].text = f"Noise Power (W): {Pn:.6e}"
    result_labels['Received Power (W)'].text = f"Received Power (W): {Pr:.6e}"
    result_labels['SNR (dB)'].text = f"SNR (dB): {SNR_dB:.2f}"
    result_labels['Channel Capacity (bps)'].text = f"Channel Capacity (bps): {H:.2e}"

def calculate_arrival_time(R_value):
    # Call the arrival_time function
    time = sim_link_budget.arrival_time(R_value)

    # update the results labels with the computed values
    result_labels['Arrival Time (s)'].text = f"Arrival Time (s): {time:.6f}"

def calculate_frequency_doppler_shift(v_value, f_value, psi):
    # Call the frequency_doppler_shift function
    fmax = sim_link_budget.frequency_doppler_shift(v_value, f_value, psi)

    # update the results labels with the computed values
    result_labels['Doppler Shifted Frequency (Hz)'].text = f"Doppler Shifted Frequency (Hz): {fmax:.2f}"

# Create the UI layout
with ui.row().style('width: 100%; justify-content: space-between'):
    # Left side of the screen for inputs
    with ui.column().style('width: 50%;'):
        #### Create the input fields ####
        ui.label("Transmit Antenna Gain (dB)")
        Gt = ui.number(value=2).props('clearable')

        ui.label("Transmit Power (W)")
        Pt = ui.number(value=1).props('clearable')

        ui.label("Transmit Frequency (Hz)")
        f = ui.number(value=920000000).props('clearable')

        ui.label("Transmit Signal Bandwidth (Hz)")
        B = ui.number(value=1000000).props('clearable')

        ui.label("Repeater Antenna Gain (dB)")
        Gr = ui.number(value=2).props('clearable')

        ui.label("Distance from Primary station to Repeater Station (m)")
        R = ui.number(value=100000).props('clearable')

        ui.label("Velocity of Spacecraft (m/s)")
        v = ui.number(value=28000000).props('clearable')

        #### Buttons to trigger calculations ####
        ui.button('Calculate Uplink', on_click=lambda: calculate_uplink(Gt.value, Pt.value, f.value, B.value, Gr.value, R.value)) 
        ui.button('Calculate Arrival Time', on_click=lambda: calculate_arrival_time(R.value)) 
        ui.button('Calcculate Doppler Shift', on_click=lambda: calculate_frequency_doppler_shift(v.value, f.value, np.pi/4))

    # Right side of the screen for the results
    with ui.column().style('width: 40%;') as right_column:
        # Placeholder for the result labels
        result_labels = {
            'Wavelength (m)': ui.label("Wavelength (m): "),
            'Noise Power (W)': ui.label("Noise Power (W): "),
            'Received Power (W)': ui.label("Received Power (W): "),
            'SNR (dB)': ui.label("SNR (dB): "),
            'Channel Capacity (bps)': ui.label("Channel Capacity (bps): "),
            'Arrival Time (s)': ui.label("Arrival Time (s): "),
            'Doppler Shifted Frequency (Hz)': ui.label("Doppler Shifted Frequency (Hz): ")
        }

# Run the NiceGUI application
ui.run()
