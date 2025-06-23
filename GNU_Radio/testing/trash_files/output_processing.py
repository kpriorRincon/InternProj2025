import numpy as np
import demodulator as dm
from scipy.signal import fftconvolve
from commpy.filters import rrcosfilter
from matplotlib import pyplot as plt
import correlator as cr

#################################### Helper Functions #############################
def bits_to_text(bits):
    """ Convert bitstream to ASCII text """
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            continue
        chars.append(chr(int(''.join(map(str, byte)), 2)))
    return ''.join(chars)

#################################### Load Received Data #######################

raw_data = np.fromfile("bits_read_in.bin", dtype=np.complex64)
print("Raw data shape:", raw_data.shape)

#################################### Correlator ###############################

start, end = cr.correlator(raw_data, 2)
correlated_data = raw_data[start:end]

##################################### Demodulation ############################

if len(correlated_data) > 0:
    bits = dm.phase_rotation_handler(correlated_data)

    print("Bits received:", bits)
    print("Total bits received: ", len(bits))

    # Convert to text
    text = bits_to_text(bits)
    print(f"Decoded message: '{text}'")
    print("Received message length: ", len(text))
else:
    print("No symbols recovered!")

# print("Received symbols:\n", symbols)
plt.scatter(np.real(raw_data), np.imag(raw_data), marker='o')
plt.grid()
plt.title("Received Symbols")
plt.show()
