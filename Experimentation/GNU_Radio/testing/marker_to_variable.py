import numpy as np
import configparser

input_file = 'modulated_marker.dat'
output_file = 'marker_config.ini'

try:
    preamble = np.fromfile(input_file, dtype=np.complex64)
except FileNotFoundError:
    print(f"File '{input_file}' not found.")
    preamble = np.array([], dtype=np.complex64)

preamble_str = ','.join(f'({c.real}+{c.imag}j)' for c in preamble)

config = configparser.ConfigParser()
config['preamble'] = {'vector': preamble_str}

with open(output_file, 'w') as configfile:
    config.write(configfile)

print(f"Preamble written to config file '{output_file}'")
