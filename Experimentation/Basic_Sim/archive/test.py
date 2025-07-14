import numpy as np
from nicegui import ui
from matplotlib import pyplot as plt
import Sig_Gen as SigGen
# with ui.matplotlib(figsize=(3, 2)).figure as fig:
#     x = np.linspace(0.0, 5.0)
#     y = np.cos(2 * np.pi * x) * np.exp(-x)
#     ax = fig.gca()
#     ax.plot(x, y, '-')

# ui.run()
sig_gen = SigGen.SigGen()
sig_gen.sample_rate = 40e9  # Set sample rate
sig_gen.freq = 906e6  # Set frequency
sig_gen.symbol_rate =  .3*sig_gen.freq # Set symbol rate
sig_gen.amp = 1.0  # Set amplitude

message = 'hello'
bit_stream = sig_gen.message_to_bits(message)
t, qpsk_waveform, t_vertical_lines, symbols = sig_gen.generate_qpsk(bit_stream)


plt.plot(t, qpsk_waveform)

plt.ylim(-1/np.sqrt(2)*sig_gen.amp-.5, 1/np.sqrt(2)*sig_gen.amp+.5)
for lines in t_vertical_lines:
    #add vertical lines at the symbol boundaries
    if lines < len(t):
        plt.axvline(x=lines, color='black', linestyle='--', linewidth=1)

        #add annotation for the symbol e.g. '00', '01', '10', '11'
        # Reverse mapping: symbol -> binary pair
        symbol = symbols[t_vertical_lines.index(lines)]
        # Reverse the mapping to get binary pair from symbol
        reverse_mapping = {v: k for k, v in sig_gen.mapping.items()}
        binary_pair = reverse_mapping.get(symbol, '')
        formatted_pair =str(binary_pair).replace("(", "").replace(")", "").replace(", ", "")
        #debug
        #print(formatted_pair)
        x_dist = 1 / (2.7 * sig_gen.symbol_rate) #half the symbol period 
        y_dist = 0.707*sig_gen.amp + .2 # 0.807 is the amplitude of the QPSK waveform
        plt.annotate(formatted_pair, xy=(lines, 0), xytext=(lines + x_dist, y_dist), fontsize=17)
plt.title(f'QPSK Waveform for {message}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
