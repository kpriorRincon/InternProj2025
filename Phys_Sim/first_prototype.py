import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater
from sim_qpsk_noisy_demod import sample_read_output
from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt

#create all of the objects

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

    # --- Time-domain plot: Original QPSK ---
    plt.subplot(3, 3, 1)
    plt.plot(t, np.real(input_qpsk))  # convert time to microseconds
    plt.title("Original QPSK Signal (Time Domain)")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1e-7)
    plt.grid(True)

    # --- Time-domain plot: Shifted QPSK ---
    plt.subplot(3, 3, 2)
    plt.plot(t, np.real(qpsk_shifted))
    plt.title("Shifted QPSK Signal (Time Domain)")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1e-7)
    plt.grid(True)
    
    plt.subplot(3, 3, 3)
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

    plt.subplot(3, 3, 4)
    plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT of QPSK After Frequency Shift")
    plt.xlim(0, fs / 2)  # From 0 to fs in MHz
    plt.ylim(0, np.max(mag_input) + 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.subplot(3, 3, 5)
    plt.plot(freqs, mag_filtered, label="Filtered QPSK", alpha=0.8)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT of QPSK After Filtering")
    plt.xlim(0, fs / 2)  # From 0 to fs in MHz
    plt.ylim(0, np.max(mag_input) + 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.subplot(3, 3, 6)
    plt.plot(freqs, mag_amp, label="Amplified QPSK", alpha=0.8)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT of QPSK After Amplification")
    plt.xlim(0, fs / 2)  # From 0 to fs in MHz
    plt.ylim(0, np.max(mag_input) + 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(3, 3, 7)
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



f_carrier = 910e6

desired_f = 960e6

def main():
    symbol_rate = 10e6
    fs_sampling = 4e9 #sample frequency
    sig_gen = Sig_Gen.SigGen()
    sig_gen.freq = f_carrier
    sig_gen.sample_rate =fs_sampling
    sig_gen.symbol_rate = symbol_rate 

    repeater = Repeater.Repeater(desired_frequency=desired_f, sampling_frequency=fs_sampling, gain=2)

    user_input = input("Enter a message to be sent: ")
    message_bits = sig_gen.message_to_bits(user_input)

    print(message_bits)
    t, qpsk, lines, symbols = sig_gen.generate_qpsk(message_bits)
    
    analytic_signal, bits = sample_read_output(qpsk,fs_sampling, symbol_rate, f_carrier)
    print(f"After generation: {bits}")

    qpsk_mixed = repeater.mix(qpsk, sig_gen.freq, t)
    #symbol_rate *= desired_f / f_carrier
    #fs_sampling *= desired_f / f_carrier
    analytic_signal, bits = sample_read_output(qpsk_mixed,fs_sampling, symbol_rate, desired_f)
    print(f"After mixing: {bits}")

    qpsk_filtered = repeater.filter(desired_f + 20e6, qpsk_mixed, order=10)
    
    analytic_signal, bits = sample_read_output(qpsk_filtered,fs_sampling, symbol_rate, desired_f)
    print(f"After filter: {bits}")

    qpsk_amp = repeater.amplify(input_signal=qpsk_filtered)
    
    plotting(t, qpsk, qpsk_mixed, qpsk_filtered, qpsk_amp, sig_gen.sample_rate)
    analytic_signal, bits = sample_read_output(qpsk_amp,fs_sampling, symbol_rate, desired_f)
    print(f"After amp: {bits}")


if __name__ == "__main__":
    main()