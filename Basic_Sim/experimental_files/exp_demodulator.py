# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, correlate
from scipy.fft import fft, ifft, fftfreq 
import Sig_Gen_Noise as SigGen
from commpy import filters 
import pickle

#####################################################
#
# Author: Trevor Wiseman
#
#####################################################

######### Global Variables #########
phase_start_sequence = np.array([-1-1j, -1-1j, 1-1j, -1+1j, 
                                 1-1j, 1-1j, -1+1j, 1+1j,
                                 1+1j, 1-1j, 1-1j, -1-1j,
                                 1-1j, -1-1j, 1+1j, -1+1j])    # gold code start 
                                                                # 11 11 10 01 
                                                                # 10 10 01 00 
                                                                # 00 10 10 11 
                                                                # 10 11 00 01
#phase_start_sequence = np.array([-1+1j, -1+1j, 1+1j, 1-1j]) # this is R in QPSK
phase_end_sequence = np.array([1+1j, 1-1j, -1+1j, 1-1j,
                               1-1j, 1+1j, 1+1j, 1-1j,
                               1+1j, -1-1j, -1-1j, -1+1j,
                               1+1j, -1+1j, 1+1j, 1-1j])   # gold code end 
                                                            # 00 10 01 10 
                                                            # 10 00 00 10 
                                                            # 00 11 11 01 
                                                            # 00 01 00 10
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

## Functions used in final implementation ##

# QPSK symbol to bit mapping
def bit_reader(symbols):
    # print("Reading bits from symbols")
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
    expected_end_sequence = ''.join(str(bit) for pair in bit_reader(phase_end_sequence) for bit in pair)
    best_bits = None                                                                                            # holds the best bits found
    #print("Expected Start Sequence: ", expected_start_sequence)                                                 # debug statement
    #print("Expected End Sequence: ", expected_end_sequence) 
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

def get_string(bits):
    """Convert bits to string."""
    
    # if len(bits) % 8 != 0:
    #     raise ValueError(f"Bit list length ({len(bits)}) must be multiple of 8")
    
    ascii_chars = []

    # Process 8 bits at a time
    for i in range(0, len(bits), 8):
        # Get 8 bits (one byte)
        byte_bits = bits[i:i+8]
        
        # Convert bits to binary string then to integer
        byte_str = ''.join(map(str, byte_bits))
        byte_value = int(byte_str, 2)
        
        # Convert to ASCII character
        ascii_chars.append(chr(byte_value))

    return ''.join(ascii_chars)

def rrc_filter(beta, N, Ts, fs):
    
    """
    Generate a Root Raised-Cosine (RRC) filter (FIR) impulse response

    Parameters:
    - beta : Roll-off factor (0 < beta <= 1)
    - N : Total number of taps in the filter (the filter span)
    - Ts : Symbol period 
    - fs : Sampling frequency/rate (Hz)

    Returns:
    - h : The impulse response of the RRC filter in the time domain
    - time : The time vector of the impulse response

    """

    # The number of samples in each symbol
    samples_per_symbol = int(fs * Ts)

    # The filter span in symbols
    total_symbols = N / samples_per_symbol

    # The total amount of time that the filter spans
    total_time = total_symbols * Ts

    # The time vector to compute the impulse response
    time = np.linspace(-total_time / 2, total_time / 2, N, endpoint=False)

    # ---------------------------- Generating the RRC impulse respose ----------------------------

    # The root raised-cosine impulse response is generated from taking the square root of the raised-cosine impulse response in the frequency domain

    # Raised-cosine filter impulse response in the time domain
    num = np.cos( (np.pi * beta * time) / (Ts) )
    denom = 1 - ( (2 * beta * time) / (Ts) ) ** 2
    g = np.sinc(time / Ts) * (num / denom)

    # Raised-cosine filter impulse response in the frequency domain
    fg = fft(g)

    # Root raised-cosine filter impulse response in the frequency domain
    fh = np.sqrt(fg)

    # Root raised-cosine filter impulse respone in the time domain
    h = ifft(fh)

    return time, h 

def buffer_correlation(bits):
    start_index = 0
    end_index = len(bits)
    start_found = False
    # Define start and end sequences
    # 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 0 0 0 1 0 1 0 1 1 1 0 1 1 0 0 0 1
    start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                      1, 0, 1, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 1, 0, 1, 1,
                      1, 0, 1, 1, 0, 0, 0, 1]
    
    # 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0 1 0 0 1 0
    end_sequence = [0, 0, 1, 0, 0, 1, 1, 0,
                    1, 0, 0, 0, 0, 0, 1, 0, 
                    0, 0, 1, 1, 1, 1, 0, 1, 
                    0, 0, 0, 1, 0, 0, 1, 0]
    for i in range(0, len(bits)):
        if bits[i:i+32] == start_sequence:
            start_index = i
            start_found = True
    
    if start_found:
        print("found start...\nlooking for end...")
        for i in range(start_index, len(bits)):
            if bits[i:i+32] == end_sequence:
                end_index = i

    return start_index, end_index

#Cross-Correlation
def cross_correlation(baseband_sig, sample_rate, symbol_rate):
    start_index = 0
    end_index = len(baseband_sig)
    
    # Define start and end sequences
    # 11111001101001000010101110110001
    start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                      1, 0, 1, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 1, 0, 1, 1,
                      1, 0, 1, 1, 0, 0, 0, 1]
    
    # 00100110100000100011110100010010
    end_sequence = [0, 0, 1, 0, 0, 1, 1, 0,
                    1, 0, 0, 0, 0, 0, 1, 0, 
                    0, 0, 1, 1, 1, 1, 0, 1, 
                    0, 0, 0, 1, 0, 0, 1, 0]
    
    # print(f"Looking for start sequence: {start_sequence}")
    # print(f"Looking for end sequence: {end_sequence}")

    sig_gen = SigGen.SigGen(0, 1.0, sample_rate, symbol_rate)
    _, start_waveform, _, _ = sig_gen.generate_qpsk(start_sequence, False, 0.1)
    _, end_waveform, _, _ = sig_gen.generate_qpsk(end_sequence, False, 0.1)

    # reduce signal strength   
    baseband_sig = (baseband_sig - baseband_sig.min()) / (baseband_sig.max() - baseband_sig.min())
    # print("Signal intensity: ", baseband_sig.max())
    
    # Correlate with start sequence
    correlated_signal = fftconvolve(baseband_sig, np.conj(np.flip(start_waveform)), mode='full')
    end_cor_signal = fftconvolve(baseband_sig, np.conj(np.flip(end_waveform)), mode='full')
    #correlated_signal = correlate(baseband_sig, np.conj(start_waveform), mode='full')
    correlated_signal = (correlated_signal -correlated_signal.min()) / (correlated_signal.max() - correlated_signal.min())
    start_waveform = (start_waveform -start_waveform.min())/ (start_waveform.max() - start_waveform.min())
    end_waveform = (end_waveform -end_waveform.min())/ (end_waveform.max() - end_waveform.min())
    
    # Find maximum correlation
    #start_index = np.argmax(np.abs(correlated_signal)) - 16*int(sample_rate/symbol_rate)
    start_index = np.argmax(np.abs(correlated_signal)) - 16*int(sample_rate/symbol_rate)
    # print("Start Index: ", start_index)
    end_index = np.argmax(np.abs(end_cor_signal))
    symbols = down_sampler(baseband_sig[start_index:end_index], sample_rate, symbol_rate)
    bits = error_handling(symbols)
    # print("Correlated bit sequence: ", bits)
    
    # Plot the best correlation result
    # plt.figure(figsize=(12, 4))

    # # plot the start correlation
    # plot_start = 0
    # print("plot start index: ", plot_start)
    # plot_end = len(correlated_signal)
    # print("plot end index: ", plot_end)
    # t = np.arange(plot_start, plot_end)
    # plt.plot(np.arange(0, len(baseband_sig)), np.abs(baseband_sig[plot_start:plot_end]), label='received signal')
    # plt.plot(np.arange(0, len(correlated_signal)), np.abs(correlated_signal[plot_start:plot_end]), '--', label='correlated signal')
    # plt.plot(np.arange(start_index, start_index+len(start_waveform)), np.abs(start_waveform), ':', label='start sequence')
    
    # # Vertical line at peak
    # plt.axvline(x=start_index, color='r', linestyle='--', label='Start Location')
    
    # plt.xlabel("Sample Index")
    # plt.ylabel("Correlation Magnitude (dB)")
    # plt.title("Cross Correlation - Start Sequence")
    # plt.legend()
    # #plt.ylim(0, 150)
    # plt.grid(True)
    
    return start_index, end_index

# sample the received signal and do error checking
def demodulator(qpsk_waveform, sample_rate, symbol_rate, t, fc):
    ## tune to baseband ##
    print("Tuning to basband...")
    baseband_sig = qpsk_waveform * np.exp(-1j * 2 * np.pi * fc * t)

    # root raised cosine matched filter
    beta = 0.35
    #_, pulse_shape = filters.rrcosfilter(300, beta, 1/symbol_rate, sample_rate)
    _, pulse_shape = rrc_filter(beta, 8, 1/symbol_rate, sample_rate)
    pulse_shape = np.convolve(pulse_shape, pulse_shape)/2
    signal = np.convolve(pulse_shape, baseband_sig, 'same')

    #find the desired signal
    # lam = 3e8 / fc  # wavelength of the carrier frequency
    # #v = 7.8e3 # average speed of a satellite in LEO
    # v = 0
    # doppler = v / lam   # calculated doppler shift
    # print("Doppler shift: ", doppler)
    # freqs = np.linspace(fc-doppler, fc+doppler, 4)
    start_index, end_index = cross_correlation(signal, sample_rate, symbol_rate)
    analytic_sig = signal[start_index:end_index]

    # sample the analytic signal
    print("Sampling the analytic signal...")
    sampled_symbols = down_sampler(analytic_sig, sample_rate, symbol_rate)

    # decode the symbols and error check the start sequence
    print("Decoding symbols and checking for start sequence...")
    best_bits = error_handling(sampled_symbols)

    ################################ Buffer Correlation Method #########################################
    # # sample the analytic signal
    # print("Sampling the analytic signal...")
    # sampled_symbols = down_sampler(signal, sample_rate, symbol_rate)

    # # get to bits
    # bit_string = error_handling(sampled_symbols)
    # print("Here are my bits: ", bit_string)
    # bits = [int(bit) for bit in bit_string]

    # # match to start and end
    # start, end = buffer_correlation(bits)
    # print("start index:", start)
    # print("end index: ", end)
    # best_bits = bits[start:end]

    return signal, sampled_symbols, best_bits

##### MAIN TEST Function #####

def main():

    # Define start and end sequences
    # 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 0 0 0 1 0 1 0 1 1 1 0 1 1 0 0 0 1
    start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                      1, 0, 1, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 1, 0, 1, 1,
                      1, 0, 1, 1, 0, 0, 0, 1]
    
    # 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0 1 0 0 1 0
    end_sequence = [0, 0, 1, 0, 0, 1, 1, 0,
                    1, 0, 0, 0, 0, 0, 1, 0, 
                    0, 0, 1, 1, 1, 1, 0, 1, 
                    0, 0, 0, 1, 0, 0, 1, 0]

    # Input message
    bad_data = " garbage garbage garbage garbage"
    message = " will this always work for checking the stuff? "
    start = get_string(start_sequence)
    end = get_string(end_sequence)
    tx_message = bad_data + start + message + end + bad_data
    print("Message:", tx_message)

    # Convert message to binary
    message_binary = ''.join(format(ord(char), '08b') for char in tx_message)
    grouped_bits = ' '.join(message_binary[i:i+2] for i in range(0, len(message_binary), 2))
    bit_sequence = [int(bit) for bit in message_binary]
    print("Binary Message:", grouped_bits)

    #Signal generation parameters
    fc = 910e6          # Carrier frequency for modulation
    sample_rate = 4e9   # sample rate
    symbol_rate = 1e6   # symbol rate
    

    # Generate QPSK waveform using your SigGen class
    print("Generating QPSK waveform...")
    sig_gen = SigGen.SigGen(fc, 1.0, sample_rate, symbol_rate)
    t, qpsk_waveform, _, _ = sig_gen.generate_qpsk(bit_sequence, False, 0.1)

    # demodulate
    analytical_output, sampled_symbols, flat_bits = demodulator(qpsk_waveform, sample_rate, symbol_rate, t, fc)




    # # read in pickle
    # import os

    # # Get the directory where your script is located
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(script_dir, 'controller_data.pkl')

    # with open(file_path, 'rb') as infile:
    #     data = pickle.load(infile)

    # rep_outgoing_signal = data['repeater outgoing signal']
    # f_out = data['freq out']
    # t = data['time']

    # sample_rate = 4e9 
    # symbol_rate = 10e6
  
    # # decode the waveform
    # print("Decoding QPSK waveform...")
    # analytical_output, sampled_symbols, flat_bits = demodulator(rep_outgoing_signal, sample_rate, symbol_rate, t, f_out)




    # Convert to ASCII characters
    original = get_string(message_binary)
    decoded = get_string(flat_bits)

    # Print the original and decoded
    print("Transmitted Bits:", original)
    print("Decoded Bits:    ", decoded)

    # if original == decoded:
    #     print("Success: bits match!")
    # else:
    #     print("Mismatch detected.")


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

if __name__=="__main__":
    main()