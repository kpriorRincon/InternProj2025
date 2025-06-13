#################
#
# Author: Trevor
#
#################
# imports
from rtlsdr import RtlSdr
import exp_demodulator as ed
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import Sig_Gen_Noise as SigGen
from scipy.stats import norm
from scipy.optimize import minimize

# What we are transmitting as a test
# 111110011010010000101011101100010000111100100110100000100011110100010010

# Define the negative log-likelihood function for a normal distribution
def neg_log_likelihood(params, data):
    mu, sigma = params
    # Avoid invalid values for sigma
    if sigma <= 0:
        return np.inf
    return -np.sum(norm.logpdf(data, loc=mu, scale=sigma))

# detection function
def detect(sig, sample_rate, symbol_rate, threshold, fc, t):
    # detection bool
    found = False
    
    # Define start and end sequences
    start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                     1, 0, 1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0, 1, 1,
                     1, 0, 1, 1, 0, 0, 0, 1]
    
    sig_gen = SigGen.SigGen(0, 1.0, sample_rate, symbol_rate)
    # Unpacking and method call
    _, start_gold, _, _ = sig_gen.generate_qpsk(start_sequence, False, 0.1)

    # tune to baseband
    sig = sig * np.exp(-1j*2*np.pi*fc*t)
    
    # Normalize the received signal
    sig_norm = (sig - np.mean(sig)) / np.std(sig)
    
    # Normalize the gold sequence
    start_gold_norm = (start_gold - np.mean(start_gold)) / np.std(start_gold)
    
    # Cross-correlation using FFT convolution
    cor_sig = fftconvolve(sig_norm, np.conj(np.flip(start_gold_norm)), mode='full')
    
    # Get the correlation magnitude
    cor_mag = np.abs(cor_sig)
    
    # Find peak correlation
    peak_correlation = np.max(cor_mag)
    peak_index = np.argmax(cor_mag)
    
    print(f"Peak Correlation: {peak_correlation:.4f}")
    print(f"Peak Index: {peak_index}")
    
    # Use a more reasonable threshold based on expected correlation values
    # For normalized signals, correlation peaks are typically in range [0, sqrt(N)]
    # where N is the length of the shorter sequence
    expected_max = np.sqrt(len(start_gold_norm))
    normalized_threshold = threshold * expected_max
    
    if peak_correlation > normalized_threshold:
        found = True
        print(f"Signal detected! Peak correlation: {peak_correlation:.4f} > threshold: {normalized_threshold:.4f}")
    else:
        print(f"No signal detected. Peak correlation: {peak_correlation:.4f} <= threshold: {normalized_threshold:.4f}")
    
    return found

# configure the RTL SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6  # Hz
sdr.center_freq = 920e6  # Hz
sdr.freq_correction = 60  # PPM
print("Valid gains:", sdr.valid_gains_db)
sdr.gain = 49.6
print("Set gain:", sdr.gain)

initial_samples = 2048 * 3
total_samples = 4 * initial_samples
symbol_rate = 1e6
start_cor = False
x = None
t = None

# Adjustable detection threshold (you may need to tune this)
detection_threshold = .85  # Adjust based on your signal characteristics

# run the detection algorithm continuously until signal is found
print("Searching for the signal...")
try:
    while start_cor == False:
        # read samples
        x = sdr.read_samples(initial_samples)
        t = np.arange(len(x)) / sdr.sample_rate  # create time vector based on actual data length
        
        # Check if we got valid samples
        if x is None or len(x) == 0:
            print("No samples received, continuing...")
            continue
            
        
        
        # detect the signal (using detection_threshold instead of mu_mle)
        start_cor = detect(x, sdr.sample_rate, symbol_rate, detection_threshold, sdr.center_freq, t)
        
        if not start_cor:
            print("Continuing search...")

    print("Signal found! \nDemodulating...")
    
    # after the signal is found demodulate and display
    signal, sampled_symbols, best_bits = ed.demodulator(x, sdr.sample_rate, symbol_rate, t, sdr.center_freq)
    print("bits: ", best_bits)
    print("Message: ", ed.get_string(best_bits))
    
    # plot the data
    # constellation plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    axs[0].scatter(np.real(sampled_symbols), np.imag(sampled_symbols))
    axs[0].grid(True)
    axs[0].set_title('Constellation Plot of Sampled Symbols')
    axs[0].set_xlabel('Real')
    axs[0].set_ylabel('Imaginary')
    
    # Plot the waveform and phase
    axs[1].plot(np.arange(0, len(signal)), np.real(signal), label='I (real part)')
    axs[1].plot(np.arange(0, len(signal)), np.imag(signal), label='Q (imag part)')
    axs[1].set_title('Filtered Baseband Signal Time Signal (Real and Imag Parts)')
    axs[1].set_xlabel('Sample Index')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid()
    axs[1].legend()
    
    # plot the fft
    ao_fft = np.fft.fft(signal)
    # Fixed: corrected the frequency calculation
    freqs = np.fft.fftfreq(len(signal), d=1/sdr.sample_rate)
    axs[2].plot(freqs, 20*np.log10(np.abs(ao_fft) + 1e-12))  # Added small value to avoid log(0)
    axs[2].set_title('FFT of the Filtered Base Band Signal')
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Magnitude (dB)')  # Fixed typo
    axs[2].grid()
    
    plt.tight_layout()
    plt.show()

except KeyboardInterrupt:
    print("\nSearch interrupted by user")
finally:
    # Clean up SDR resources
    sdr.close()
    print("SDR closed")
