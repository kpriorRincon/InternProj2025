from nicegui import ui
import plotly.graph_objects as go
import sim_qpsk_noisy_read as sqnr

# Store the QPSK symbols in a global variable
generated_symbols = None

# Function to calculate and display results
def generate_rand_symbols(num_symbols):
    global generated_symbols  # Use the global variable to store symbols

    # Call the compute_parameters function
    generated_symbols = sqnr.random_symbol_generator(num_symbols)

    # Create and display the plot
    fig = go.Figure(data=go.Scatter(x=generated_symbols.real, y=generated_symbols.imag, mode='markers', marker=dict(size=5)))
    ui.plotly(fig)  # This displays the Plotly figure

def add_noise_to_symbols(noise_power, num_symbols):
    global generated_symbols  # Use the global variable to access the symbols

    if generated_symbols is None:
        print("Please generate symbols first!")
        return

    # Call the compute_parameters function
    noisy_symbols = sqnr.noise_adder(generated_symbols, noise_power, num_symbols)

    # Create and display the noisy plot
    fig = go.Figure(data=go.Scatter(x=noisy_symbols.real, y=noisy_symbols.imag, mode='markers', marker=dict(size=5)))
    ui.plotly(fig)  # This displays the noisy Plotly figure

def read_bits_from_symbols():
    global generated_symbols  # Use the global variable to access the symbols

    if generated_symbols is None:
        print("Please generate symbols first!")
        return

    # Call the compute_parameters function
    bits = sqnr.bit_reader(generated_symbols)

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
        ui.button('Generate Symbols', on_click=lambda: generate_rand_symbols(num_symbols.value))
        ui.button('Add Noise', on_click=lambda: add_noise_to_symbols(noise_power.value, num_symbols.value)) 
        ui.button('Generate Bits', on_click=lambda: read_bits_from_symbols())

    # Right side of the screen for the results
    with ui.column().style('width: 40%;') as right_column:
        # Placeholder for the result labels
        result_labels = {
            'Bit Sequence: ': ui.label("Bit Sequence: "),
        }

# Run the NiceGUI application
ui.run()
