# imports
import numpy as np
import matplotlib.pyplot as plt
import Sig_Gen as SigGen
from scipy.signal import hilbert

######### Global Variables #########
phase_start_sequence = np.array([45, 135, 225, 135])

######## Functions ########

# Generate random QPSK symbols (for testing)
def random_symbol_generator(num_symbols=100):
    x_int = np.random.randint(0, 4, num_symbols)
    x_degrees = x_int * 360 / 4.0 + 45
    x_radians = x_degrees * np.pi / 180.0
    x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians)
    return x_symbols

# Add noise to symbols
def noise_adder(x_symbols, noise_power=0.1, num_symbols=100):
    n = (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)) / np.sqrt(2)
    phase_noise = np.random.randn(len(x_symbols)) * 0.1
    r = x_symbols * np.exp(1j * phase_noise) + n * np.sqrt(noise_power)
    return r

# QPSK symbol to bit mapping
def bit_reader(symbols):
    bits = np.zeros((len(symbols), 2), dtype=int)
    for i in range(len(symbols)):
        angle = np.angle(symbols[i], deg=True) % 360

        # don't know why, but this is the only way to get the angles in the right range.
        # Might be because of the Hilbert transform?
        if 0 <= angle < 90:
            bits[i] = [1, 1]  # 45°
        elif 90 <= angle < 180:
            bits[i] = [1, 0]  # 135°
        elif 180 <= angle < 270:
            bits[i] = [0, 0]  # 225°
        else:
            bits[i] = [0, 1]  # 315°
    return bits

def sample_read_output(qpsk_waveform, sample_rate, symbol_rate):
    # compute the Hilbert transform
    analytic_signal = hilbert(qpsk_waveform)

    # Sample at symbol midpoints
    samples_per_symbol = int(sample_rate / symbol_rate)
    offset = samples_per_symbol // 2
    sampled_symbols = analytic_signal[offset::samples_per_symbol]

    # Optional normalization to unit magnitude
    sampled_symbols /= np.abs(sampled_symbols)

    # Decode bits
    decoded_bits = bit_reader(sampled_symbols)

    # Flatten bits to string
    flat_bits = ''.join(str(b) for pair in decoded_bits for b in pair)

    return analytic_signal, flat_bits

##### MAIN TEST #####

# Input message
message = "test"
print("Message:", message)

# Convert message to binary
message_binary = ''.join(format(ord(char), '08b') for char in message)
grouped_bits = ' '.join(message_binary[i:i+2] for i in range(0, len(message_binary), 2))
bit_sequence = [int(bit) for bit in message_binary]
print("Binary Message:", grouped_bits)

# Add noise to the waveform
#noisy_bits = noise_adder(bit_sequence, noise_power=0.1, num_symbols=len(bit_sequence)/2)

# Signal generation parameters
sample_rate = 1e6  # 1 MHz
symbol_rate = 1000  # 1 kHz

# Generate QPSK waveform using your SigGen class
sig_gen = SigGen.SigGen(1000, 1.0, sample_rate, symbol_rate)
t, qpsk_waveform,t_vertical_lines, symbols = sig_gen.generate_qpsk(bit_sequence)

# decode the waveform
# apply hilbert transform
analytical_output, flat_bits = sample_read_output(qpsk_waveform, sample_rate, symbol_rate)

# Convert to ASCII characters
decoded_chars = [chr(int(flat_bits[i:i+8], 2)) for i in range(0, len(flat_bits), 8)]
decoded_message = ''.join(decoded_chars)
print("Decoded Message:", decoded_message)

original = ' '.join(message_binary[i:i+2] for i in range(0, len(message_binary), 2))
decoded  = ' '.join(flat_bits[i:i+2] for i in range(0, len(flat_bits), 2))

print("Transmitted Bits:", original)
print("Decoded Bits:    ", decoded)

if original == decoded:
    print("Success: bits match!")
else:
    print("Mismatch detected.")


# Plot the waveform and phase
plt.figure(figsize=(10, 4))
plt.plot(t, analytical_output.real, label='I (real part)')
plt.plot(t, analytical_output.imag, label='Q (imag part)')
plt.title('Hilbert Transformed Waveform (Real and Imag Parts)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()