# imports
import numpy as np
import matplotlib.pyplot as plt
import Sig_Gen as SigGen

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

        if 0 <= angle < 90:
            bits[i] = [0, 0]  # 45°
        elif 90 <= angle < 180:
            bits[i] = [0, 1]  # 135° (was [0, 1])
        elif 180 <= angle < 270:
            bits[i] = [1, 1]  # 225° (was [1, 1])
        else:
            bits[i] = [1, 0]  # 315°
    return bits



##### MAIN TEST #####

# Input message
message = "ABCD"
print("Message:", message)

# Convert message to binary
message_binary = ''.join(format(ord(char), '08b') for char in message)
grouped_bits = ' '.join(message_binary[i:i+2] for i in range(0, len(message_binary), 2))
bit_sequence = [int(bit) for bit in message_binary]
print("Binary Message:", grouped_bits)

# Signal generation parameters
sample_rate = 1e6  # 1 MHz
symbol_rate = 1000  # 1 kHz

# Generate QPSK waveform using your SigGen class
sig_gen = SigGen.SigGen(freq=1000, amp=1.0)
t, qpsk_waveform = sig_gen.generate_qpsk(bit_sequence, sample_rate, symbol_rate)
# Sample at symbol midpoints
Ts = int(sample_rate / symbol_rate)
sampled_symbols = qpsk_waveform[::Ts]

# Optional normalization to unit magnitude
sampled_symbols /= np.abs(sampled_symbols)  

# Decode bits
decoded_bits = bit_reader(sampled_symbols)

# Format output
flat_bitstream = ''.join(str(b) for pair in decoded_bits for b in pair)
grouped_bits = ' '.join(flat_bitstream[i:i+2] for i in range(0, len(flat_bitstream), 2))

# Output
print("Decoded Bits (grouped):", grouped_bits)

# Convert to characters
decoded_chars = [chr(int(flat_bitstream[i:i+8], 2)) for i in range(0, len(flat_bitstream), 8)]
decoded_message = ''.join(decoded_chars)
print("Decoded Message:", decoded_message)

original = ' '.join(message_binary[i:i+2] for i in range(0, len(message_binary), 2))
decoded  = ' '.join(flat_bitstream[i:i+2] for i in range(0, len(flat_bitstream), 2))

print("Transmitted Bits:", original)
print("Decoded Bits:    ", decoded)

if original == decoded:
    print("Success: bits match!")
else:
    print("Mismatch detected.")


# Plot the waveform and phase
plt.figure(figsize=(10, 4))
plt.plot(t, qpsk_waveform.real, label='I (real part)')
plt.plot(t, qpsk_waveform.imag, label='Q (imag part)')
plt.title('QPSK Waveform (Real and Imag Parts)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
