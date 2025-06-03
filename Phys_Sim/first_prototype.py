import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater
#from sim_qpsk_noisy_demod import sample_read_output
from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt

#create all of the objects



######### Global Variables #########
phase_start_sequence = np.array([-1+1j, -1+1j, 1+1j, 1-1j]) # this is the letter R in QPSK
phases = np.array([45, 135, 225, 315])  # QPSK phase angles in degrees



# QPSK symbol to bit mapping
def bit_reader(symbols):
    bits = np.zeros((len(symbols), 2), dtype=int)
    for i in range(len(symbols)):
        angle = np.angle(symbols[i], deg=True) % 360

        # don't know why, but this is the only way to get the angles in the right range.
        # Might be because of the Hilbert transform?
        if 0 <= angle < 90:
            bits[i] = [0, 0]  # 45°
        elif 90 <= angle < 180:
            bits[i] = [0, 1]  # 135°
        elif 180 <= angle < 270:
            bits[i] = [1, 1]  # 225°
        else:
            bits[i] = [1, 0]  # 315°
    return bits

def matched_filter(input_signal, fc):
    # create template based on the carrier frequency
    template = np.exp(1j * 2 * np.pi * fc * np.arange(len(input_signal)) / len(input_signal))

    ## Perform the auto-correlation by convolving the input signal with the flipped template ##

    # flip
    template_flipped = np.flip(template)
    
    # fft
    input_signal_fft = np.fft.fft(input_signal)
    template_flipped_fft = np.fft.fft(template_flipped)

    # convolve
    convolved_fft = input_signal_fft * template_flipped_fft
    convolved = np.fft.ifft(convolved_fft)

    # normalize
    convolved /= np.linalg.norm(template_flipped)

    return convolved


def sample_read_output(qpsk_waveform, sample_rate, symbol_rate):
    ## compute the Hilbert transform ##
    analytic_signal = hilbert(qpsk_waveform)    # hilbert transformation

    #analytic_signal = matched_filter(analytic_signal, fc)

    ## Sample at symbol midpoints ##
    samples_per_symbol = int(sample_rate / symbol_rate)             # number of samples per symbol
    offset = samples_per_symbol // 2                                # offset to sample at the midpoint of each symbol   
    sampled_symbols = analytic_signal[offset::samples_per_symbol]   # symbols sampled from the analytical signal
    sampled_symbols /= np.abs(sampled_symbols)                      # normalize the symbols

    ## look for the start sequence ##
    expected_start_sequence = ''.join(str(bit) for pair in bit_reader(phase_start_sequence) for bit in pair)    # put the start sequence into a string
    best_bits = None                                                                                            # holds the best bits found
    #print("Expected Start Sequence: ", expected_start_sequence)                                                 # debug statement
    og_sampled_symbols = ''.join(str(bit) for pair in bit_reader(sampled_symbols) for bit in pair)              # original sampled symbols in string format
    #print("Original sampled bits: ", og_sampled_symbols)                                                        # debug statement

    ## Loop through possible phase shifts ##
    for i in range(0, 3):   # one for each quadrant (0°, 90°, 180°, 270°)
        # Rotate the flat bits to match the start sequence
        rotated_bits = sampled_symbols * np.exp(-1j* np.deg2rad(i*90))  # Rotate by 0, 90, 180, or 270 degrees
        
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

    return analytic_signal, best_bits

def plotting(t, input_qpsk, qpsk_shifted, qpsk_filtered, qpsk_amp, fs):
    """
    Plots the original and shifted QPSK signals.

    Parameters:
    - t: Time vector for the signal.
    - input_qpsk: The original QPSK signal.
    - qpsk_shifted: The shifted QPSK signal.
    """
    # Compute FFT
    n = len(t)
    freqs = np.fft.fftfreq(n, d=1/fs)

    # FFT of original and shifted signals
    fft_input = np.fft.fft(input_qpsk)
    fft_shifted = np.fft.fft(qpsk_shifted)
    fft_filtered = np.fft.fft(qpsk_filtered)
    fft_amp = np.fft.fft(qpsk_amp)
    # Convert magnitude to dB
    mag_input = 20 * np.log10(np.abs(fft_input))
    mag_shifted = 20 * np.log10(np.abs(fft_shifted))
    mag_filtered = 20 * np.log10(np.abs(fft_filtered))
    mag_amp = 20 * np.log10(np.abs(fft_amp))
    plt.figure(figsize=(12, 10))
    plt.plot(t, np.real(input_qpsk), label="Original QPSK")
    plt.plot(t, np.real(qpsk_shifted), label="shifted QPSK")
    plt.plot(t, np.real(qpsk_filtered), label="filtered QPSK")
    plt.legend()

    
    plt.title("Combined")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 0.1e-7)
    plt.grid(True)
    plt.show()
    return
"""
    # --- Time-domain plot: Original QPSK ---
    plt.subplot(2, 3, 1)
    plt.plot(t, np.real(input_qpsk))  # convert time to microseconds
    plt.title("Original QPSK Signal (Time Domain)")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1e-7)
    plt.grid(True)

    # --- Time-domain plot: Shifted QPSK ---
    plt.subplot(2, 3, 2)
    plt.plot(t, np.real(qpsk_shifted))
    plt.title("Shifted QPSK Signal (Time Domain)")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1e-7)
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(freqs, mag_input, label="Original QPSK", alpha=0.8)
    #plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT of QPSK Before Frequency Shift")
    plt.xlim(0, fs / 2)  # From 0 to fs in MHz
    plt.ylim(0, np.max(mag_input) + 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 3, 4)
    plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT of QPSK After Frequency Shift")
    plt.xlim(0, fs / 2)  # From 0 to fs in MHz
    plt.ylim(0, np.max(mag_input) + 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 3, 5)
    plt.plot(freqs, mag_filtered, label="Filtered QPSK", alpha=0.8)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT of QPSK After Filtering")
    plt.xlim(0, fs / 2)  # From 0 to fs in MHz
    plt.ylim(0, np.max(mag_input) + 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 3, 6)
    plt.plot(freqs, mag_amp, label="Amplified QPSK", alpha=0.8)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT of QPSK After Amplification")
    plt.xlim(0, fs / 2)  # From 0 to fs in MHz
    plt.ylim(0, np.max(mag_input) + 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return

"""

f_carrier = 910e6

desired_f = 960e6

def main():
    symbol_rate = 800e6
    fs_sampling = 100e9 #sample frequency
    sig_gen = Sig_Gen.SigGen()
    sig_gen.freq = f_carrier
    sig_gen.sample_rate =fs_sampling
    sig_gen.symbol_rate = symbol_rate 

    repeater = Repeater.Repeater(desired_frequency=desired_f, sampling_frequency=fs_sampling, gain=2)

    user_input = input("Enter a message to be sent: ")
    message_bits = sig_gen.message_to_bits(user_input)

    print(message_bits)
    t, qpsk, lines, symbols = sig_gen.generate_qpsk(message_bits)
    
    analytic_signal, bits = sample_read_output(qpsk,fs_sampling, symbol_rate)
    print(f"After generation: {bits}")

    qpsk_mixed = repeater.mix(qpsk, sig_gen.freq, t)
    symbol_rate *= desired_f / f_carrier
    fs_sampling *= desired_f / f_carrier
    analytic_signal, bits = sample_read_output(qpsk_mixed,fs_sampling, symbol_rate)
    print(f"After mixing: {bits}")

    qpsk_filtered = repeater.filter(desired_f + 20e6, qpsk_mixed, order=10)
    
    analytic_signal, bits = sample_read_output(qpsk_filtered,fs_sampling, symbol_rate)
    print(f"After filter: {bits}")

    qpsk_amp = repeater.amplify(input_signal=qpsk_filtered)
    
    #plotting(t, qpsk, qpsk_mixed, qpsk_filtered, qpsk_amp, fs_sampling)
    analytic_signal, bits = sample_read_output(qpsk_amp,fs_sampling, symbol_rate)
    print(f"After amp: {bits}")


if __name__ == "__main__":
    main()