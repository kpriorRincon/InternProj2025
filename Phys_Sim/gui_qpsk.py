from nicegui import ui
import plotly.graph_objects as go
import sim_qpsk_noisy_read as sqnr

#################################################
#
#   Author: Trevor Wiseman
#
#################################################

# Function to calculate and display results
def generate_rand_symbols(num_symbols):
    # Call the compute_parameters function
    x_symbols = sqnr.random_symbol_generator(num_symbols)

    # Create and display the plot
    fig = go.Figure(data=go.Scatter(x=x_symbols.real, y=x_symbols.imag, mode='markers', marker=dict(size=5)))
    ui.plotly(fig)  # This displays the Plotly figure

def add_noise_to_symbols(x_symbols, noise_power, num_symbols):
    # Call the compute_parameters function
    r = sqnr.noise_adder(x_symbols, noise_power, num_symbols)

    # Create and display the plot
    fig = go.Figure(data=go.Scatter(x=r.real, y=r.imag, mode='markers', marker=dict(size=5)))
    ui.plotly(fig)  # This displays the Plotly figure

def read_bits_from_symbols(x_symbols):
    # Call the compute_parameters function
    bits = sqnr.bit_reader(x_symbols)

    # Update the label with the bit sequence
    result_labels['Bit Sequence: '].text = f"Bit Sequence: {bits}"

# Create the UI layout
with ui.row().style('width: 100%; justify-content: space-between'):
    # Left side of the screen for inputs
    with ui.column().style('width: 50%;'):
        #### Create the input fields ####
        ui.label("Enter number of symbols")
        num_symbols = ui.number(value=100).props('clearable')
        ui.label("Enter noise power")
        noise_power = ui.number(value=0.01).props('clearable')

        #### Buttons to trigger calculations ####
        x_symbols = ui.button('Generate Symbols', on_click=lambda: generate_rand_symbols(num_symbols.value))
        r = ui.button('Add Noise', on_click=lambda: add_noise_to_symbols(x_symbols, noise_power.value, num_symbols.value))
        ui.button('Generate Bits', on_click=lambda: read_bits_from_symbols(x_symbols))

    # Right side of the screen for the results
    with ui.column().style('width: 40%;') as right_column:
        # Placeholder for the result labels
        result_labels = {
            'Bit Sequence: ': ui.label("Bit Sequence: "),
        }

# Run the NiceGUI application
ui.run()
