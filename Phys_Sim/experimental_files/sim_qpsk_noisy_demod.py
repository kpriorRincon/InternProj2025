# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import Sig_Gen_Noise as SigGen
from commpy import filters 

#####################################################
#
# Author: Trevor Wiseman
#
#####################################################

######### Global Variables #########
phase_start_sequence = np.array([1+1j, 1-1j, 1+1j, -1+1j]) # this is the letter ! in QPSK 00100001
phase_end_sequence = np.array([1+1j, 1-1j, -1-1j, -1-1j]) # / 00101111
phases = np.array([45, 135, 225, 315])  # QPSK phase angles in degrees

######## Functions ########

## Functions for testing functionality ##
# Generate random QPSK symbols (for testing)
def random_symbol_generator(num_symbols=100):
    x_int = np.random.randint(0, 4, num_symbols)
    x_degrees = x_int * 360 / 4.0 + 45
    x_radians = x_degrees * np.pi / 180.0
    x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians)
    return x_symbols

# Add noise to symbols (for testing)
def noise_adder(x_symbols, noise_power=0.1, num_symbols=100):
    n = (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)) / np.sqrt(2)
    phase_noise = np.random.randn(len(x_symbols)) * 0.1
    r = x_symbols * np.exp(1j * phase_noise) + n * np.sqrt(noise_power)
    return r

# find the minimum
def find_start_from_minimum_before_peak(correlation_signal, search_window=100):
    corr_mag = np.abs(correlation_signal)
    peak_idx = np.argmax(corr_mag)
    
    # Search for minimum before peak
    search_start = max(0, peak_idx - search_window)
    search_end = peak_idx
    
    if search_start >= search_end:
        return peak_idx
    
    search_region = corr_mag[search_start:search_end]
    local_min_idx = np.argmin(search_region)
    return search_start + local_min_idx

# Cross-Correlation
def cross_correlation(baseband_sig, freqs, sample_rate, symbol_rate, fc):
    start_index = 0
    end_index = len(baseband_sig)  # Default to full signal
    current_max = 0
    best_start_corr = None
    best_end_corr = None
    best_start_idx = 0
    best_end_idx = len(baseband_sig)
    
    # Define start and end sequences
    start = '!'
    message_binary = ''.join(format(ord(char), '08b') for char in start)
    start_sequence = [int(bit) for bit in message_binary]
    
    end = '/'
    message_binary = ''.join(format(ord(char), '08b') for char in end)
    end_sequence = [int(bit) for bit in message_binary]
    
    print(f"Looking for start sequence: {start_sequence}")
    print(f"Looking for end sequence: {end_sequence}")
    
    # Test different frequencies (for Doppler compensation)
    for i, f in enumerate(freqs):
        print(f"Testing frequency {i+1}/{len(freqs)}: {f/1e6:.1f} MHz")
        
        # Generate reference sequences at baseband
        sig_gen = SigGen.SigGen(0, 1.0, sample_rate, symbol_rate)  # Generate at baseband (0 Hz)
        _, start_gold, _, _ = sig_gen.generate_qpsk(start_sequence, False, 0)
        _, end_gold, _, _ = sig_gen.generate_qpsk(end_sequence, False, 0)
        
        # Apply frequency offset compensation to baseband signal
        freq_offset = f - fc
        compensated_sig = baseband_sig * np.exp(-1j * 2 * np.pi * freq_offset * np.arange(len(baseband_sig)) / sample_rate)
        
        # Correlate with start sequence
        cor_sig_start = correlate(compensated_sig, np.conj(start_gold), mode='valid')
        
        # Find maximum correlation
        max_corr_abs = np.amax(np.abs(cor_sig_start))
        
        if max_corr_abs > current_max:
            print(f"Stronger correlation found at {f/1e6:.1f} MHz")
            print(f"Correlation strength: {20*np.log10(max_corr_abs):.1f} dB")
            
            current_max = max_corr_abs
            
            # Find start index (in original signal coordinates)
            start_idx_in_corr = find_start_from_minimum_before_peak(cor_sig_start, search_window=100)
            best_start_idx = start_idx_in_corr  # For 'valid' mode, this is direct
            
            # Correlate with end sequence
            cor_sig_end = correlate(compensated_sig, np.conj(end_gold), mode='valid')
            end_idx_in_corr = find_start_from_minimum_before_peak(cor_sig_end, search_window=100)
            best_end_idx = end_idx_in_corr + len(end_gold)  # Add length to get end position
            
            # Ensure end_index is not beyond signal length
            best_end_idx = min(best_end_idx, len(baseband_sig))
            
            print(f"Start index: {best_start_idx}")
            print(f"End index: {best_end_idx}")
            
            # Store best correlations for debugging
            best_start_corr = cor_sig_start
            best_end_corr = cor_sig_end
    
    # Set final indices
    start_index = best_start_idx
    end_index = best_end_idx
    
    # Final validation
    if start_index >= end_index:
        print("Warning: start_index >= end_index, using default values")
        start_index = 0
        end_index = len(baseband_sig)
    
    # Plot the best correlation result
    if best_start_corr is not None:
        plt.figure(figsize=(12, 4))
        corr_db = 20*np.log10(np.abs(best_start_corr))
        plt.plot(np.arange(len(best_start_corr)), corr_db)
        
        # Vertical line at peak
        peak_idx = np.argmax(np.abs(best_start_corr))
        plt.axvline(x=peak_idx, color='r', linestyle='--', label='Peak Location')
        
        # Horizontal lines for reference levels
        max_corr_db = np.amax(corr_db)
        plt.axhline(y=max_corr_db, color='g', linestyle='--', alpha=0.7, label='Peak Level')
        plt.axhline(y=max_corr_db - 3, color='orange', linestyle='--', alpha=0.7, label='-3dB')
        
        plt.xlabel("Sample Index")
        plt.ylabel("Correlation Magnitude (dB)")
        plt.title("Cross Correlation - Start Sequence")
        plt.legend()
        plt.ylim(65, 85)
        plt.grid(True)
    


    return start_index, end_index

## Functions used in final implementation ##

# QPSK symbol to bit mapping
def bit_reader(symbols):
    print("Reading bits from symbols")
    bits = np.zeros((len(symbols), 2), dtype=int)
    for i in range(len(symbols)):
        angle = np.angle(symbols[i], deg=True) % 360

        # codex mapping phase to bits
        if 0 <= angle < 90:
            bits[i] = [0, 0]  # 45°
        elif 90 <= angle < 180:
            bits[i] = [0, 1]  # 135°
        elif 180 <= angle < 270:
            bits[i] = [1, 1]  # 225°
        else:
            bits[i] = [1, 0]  # 315°
    return bits

# Error checking for the start sequence using a matched filter
def error_handling(sampled_symbols):
    #print("Error checking")
    ## look for the start sequence ##
    expected_start_sequence = ''.join(str(bit) for pair in bit_reader(phase_start_sequence) for bit in pair)    # put the start sequence into a string
    best_bits = None                                                                                            # holds the best bits found
    #print("Expected Start Sequence: ", expected_start_sequence)                                                 # debug statement
    og_sampled_symbols = ''.join(str(bit) for pair in bit_reader(sampled_symbols) for bit in pair)                 # original sampled symbols in string format
    #print("Sampled bits: ", og_sampled_symbols)                                                                    # debug statement

    ## Loop through possible phase shifts ##
    for i in range(0, 7):   # one for each quadrant (0°, 90°, 180°, 270°)
        # Rotate the flat bits to match the start sequence
        rotated_bits = sampled_symbols * np.exp(-1j* np.deg2rad(i*45))  # Rotate by 0, 90, 180, or 270 degrees
        
        # decode the bits
        decode_bits = bit_reader(rotated_bits)                                  # decode the rotated bits
        flat_bits = ''.join(str(bit) for pair in decode_bits for bit in pair)   # put the bits into a string
        #print("Rotated bits: ", flat_bits)                                      # debug statement
        
         # Check for presence of the known start sequence (first few symbols)
        if expected_start_sequence == flat_bits[0:8]:                   # check only first 8 symbols worth (16 bits)
            print(f"Start sequence found with phase shift: {i*90}°")
            best_bits = flat_bits                                       # store the best bits found
            break
    
    # Error state if no start sequence was found
    if best_bits is None:
        print("Start sequence not found. Defaulting to 0°")
        rotated_symbols = sampled_symbols
        decoded_bits = bit_reader(rotated_symbols)
        best_bits = ''.join(str(b) for pair in decoded_bits for b in pair)
    
    return best_bits

def down_sampler(sig, sample_rate, symbol_rate):
    # write some downsampling here
    samples_per_symbol = int(sample_rate/symbol_rate)
    symbols = sig[::samples_per_symbol]
    return symbols

# sample the received signal and do error checking
def demodulator(qpsk_waveform, sample_rate, symbol_rate, t, fc):
    ## tune to baseband ##
    print("Tuning to basband...")
    baseband_sig = qpsk_waveform * np.exp(-1j * 2 * np.pi * fc * t)

    # find the desired signal
    # lam = 3e8 / fc  # wavelength of the carrier frequency
    # v = 7.8e3 # average speed of a satellite in LEO
    # doppler = v / lam   # calculated doppler shift
    # print("Doppler shift: ", doppler)
    # freqs = np.linspace(fc-doppler, fc+doppler, 4)
    # start_index, end_index = cross_correlation(baseband_sig, freqs, sample_rate, symbol_rate, fc)
    # analytic_sig = baseband_sig[start_index:end_index]

    # root raised cosine matched filter
    beta = 0.3
    _, pulse_shape = filters.rrcosfilter(300, beta, 1/symbol_rate, sample_rate)
    pulse_shape = np.convolve(pulse_shape, pulse_shape)/2
    signal = np.convolve(pulse_shape, baseband_sig, 'same')

    # plots to see the constellations before and after tuning and after the matched filter
    # fig, axs = plt.subplots(2,1)
    # axs[0].scatter(np.real(qpsk_waveform), np.imag(qpsk_waveform))
    # axs[0].set_xlabel('Real')
    # axs[0].set_ylabel('Imaginary')
    # axs[0].set_title("Raw QPSK")
    # axs[0].grid()

    # axs[1].scatter(np.real(baseband_sig), np.imag(baseband_sig))
    # axs[1].set_title("Tuned Signal")
    # axs[1].set_xlabel('Real')
    # axs[1].set_ylabel('Imaginary')
    # axs[1].grid()

    # axs[2].scatter(np.real(signal), np.imag(signal))
    # axs[2].set_title("Analytic Signal")
    # axs[2].set_xlabel('Real')
    # axs[2].set_ylabel('Imaginary')
    # axs[2].grid()
    
    # sample the analytic signal
    print("Sampling the analytic signal...")
    sampled_symbols = down_sampler(signal, sample_rate, symbol_rate)

    # decode the symbols and error check the start sequence
    print("Decoding symbols and checking for start sequence...")
    best_bits = error_handling(sampled_symbols)

    return signal, sampled_symbols, best_bits

##### MAIN TEST Function #####

def main():
    # Input message
    message = "!ABCDE/"
    print("Message:", message)

    # Convert message to binary
    message_binary = ''.join(format(ord(char), '08b') for char in message)
    grouped_bits = ' '.join(message_binary[i:i+2] for i in range(0, len(message_binary), 2))
    bit_sequence = [int(bit) for bit in message_binary]
    print("Binary Message:", grouped_bits)

    # Signal generation parameters
    freq = 900e6            # Carrier frequency for modulation
    sample_rate = 5 * freq  # 5 times the carrier frequency for oversampling
    symbol_rate = 1e6      # 10 MHz

    # Generate QPSK waveform using your SigGen class
    print("Generating QPSK waveform...")
    sig_gen = SigGen.SigGen(freq, 1.0, sample_rate, symbol_rate)
    t, qpsk_waveform, _, _ = sig_gen.generate_qpsk(bit_sequence, False, 0.1)

    # plt.plot(t, np.imag(qpsk_waveform))
    # plt.xlim(0,50/sample_rate)
    # plt.show()

    
    # decode the waveform
    # apply hilbert transform
    print("Decoding QPSK waveform...")
    analytical_output, sampled_symbols, flat_bits = demodulator(qpsk_waveform, sample_rate, symbol_rate, t, freq)

    # Convert to ASCII characters
    decoded_chars = [chr(int(flat_bits[i:i+8], 2)) for i in range(0, len(flat_bits), 8)]
    decoded_message = ''.join(decoded_chars)
    print("Decoded Message:", decoded_message)

    # Print the originalcross_correlation
    original = ' '.join(message_binary[i:i+2] for i in range(0, len(message_binary), 2))
    decoded  = ' '.join(flat_bits[i:i+2] for i in range(0, len(flat_bits), 2))

    print("Transmitted Bits:", original)
    print("Decoded Bits:    ", decoded)

    if original == decoded:
        print("Success: bits match!")
    else:
        print("Mismatch detected.")


    # constellation plot
    fig, axs = plt.subplots(3,1)
    axs[0].scatter(np.real(sampled_symbols), np.imag(sampled_symbols))
    axs[0].grid(True)
    axs[0].set_title('Constellation Plot of Sampled Symbols')
    axs[0].set_xlabel('Real')
    axs[0].set_ylabel('Imaginary')

    # Plot the waveform and phase
    axs[1].plot(np.arange(0,len(analytical_output)), np.real(analytical_output), label='I (real part)')
    axs[1].plot(np.arange(0, len(analytical_output)), np.imag(analytical_output), label='Q (imag part)')
    axs[1].set_title('Filtered Baseband Signal Time Signal (Real and Imag Parts)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid()
    axs[1].legend()

    # plot the fft
    ao_fft = np.fft.fft(analytical_output)
    freqs = np.fft.fftfreq(len(analytical_output), d=1/2*sample_rate)
    axs[2].plot(freqs, 20*np.log10(ao_fft))
    axs[2].set_title('FFT of the Filtered Base Band Signal')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Madgnitude (dB)')
    axs[2].grid()

    plt.show()


