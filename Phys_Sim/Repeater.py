import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def find_peak(positive_mags, positive_freq_values):
    # Get indices of the 4 largest peaks
    top4_indices = np.argpartition(positive_mags, -2)[-2:]
    # Sort them by magnitude (descending)
    top4_sorted = top4_indices[np.argsort(positive_mags[top4_indices])[::-1]]

    # Extract the corresponding frequencies
    top4_freqs = positive_freq_values[top4_sorted]
    top4_mags = positive_mags[top4_sorted]

    # Choose the frequency in the middle (median)
    middle_freq = np.median(top4_freqs)

    # Optional: pick magnitude at the closest frequency to median
    closest_idx = np.argmin(np.abs(positive_freq_values - middle_freq))
    peak_mag = positive_mags[closest_idx]

    # Result
    peak_freq = middle_freq

    return peak_freq, peak_mag

class Repeater:
    def __init__(self, sampling_frequency, symbol_rate):
        self.desired_frequency = None  # Default frequency set to 1 GHz
        self.sampling_frequency = sampling_frequency
        self.symbol_rate = symbol_rate
        self.gain = None
        

        self.qpsk_mixed = None
        self.qpsk_filtered = None

    def mix(self, qpsk_signal, qpsk_frequency, t):
        """
        Mixes the input signal with a carrier frequency.

        Parameters:
        - signal: The input signal to be mixed.
        - qpsk_frequency: The frequency of the QPSK signal.
        - t: Time vector for the signal.

        Returns:
        - The mixed signal.
        """
        # Implement mixing logic here
        #Complex sinusoid (real-world)
        #mixing_signal = np.cos(2 * np.pi * (self.desired_freqeuncy + qpsk_frequency) * t)

        
        #Complex exponential (Ideal)
        mixing_signal = np.exp(1j * 2 * np.pi * (self.desired_frequency - qpsk_frequency) * t)

        # Mix the QPSK signal with the complex exponential to shift its frequency
        qpsk_shifted = qpsk_signal * mixing_signal

        return qpsk_shifted

    def filter(self, cuttoff_frequency, mixed_qpsk, order=5):
        """
        Filters the mixed signal to remove unwanted frequencies.

        Returns:
        - The filtered signal.
        """
        # Implement filtering logic here


        b, a = signal.butter(order, cuttoff_frequency, btype='low', fs=self.sampling_frequency) # butterworth filter coefficients

        # Apply filter
        filtered_sig = signal.filtfilt(b, a, mixed_qpsk)   # filtered signal
        
        return filtered_sig

    def amplify(self, input_signal):
        """
        Amplifies the signal by a specified gain.
        Parameters:
        - gain: The gain factor to amplify the signal.
        Returns:
        - The amplified signal.
        """
        # Implement amplification logic here
        return self.gain*input_signal
    
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

        peak_index = np.argmax(mag_input)
        peak_freq = freqs[peak_index]
        peak_mag = mag_input[peak_index]
        plt.plot(freqs, mag_input, label="Original QPSK", alpha=0.8)
        plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq:.2f} GHz')
        plt.text(peak_freq, peak_mag + 5, f'{peak_freq:.2f} GHz', color='r', ha='center')
        #plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("FFT of Incoming QPSK")
        plt.xlim(0, fs / 2)  # From 0 to fs in MHz
        plt.ylim(0, np.max(mag_input) + 10)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.subplot(2, 3, 4)
        peak_index = np.argmax(mag_shifted)
        peak_freq = freqs[peak_index]
        peak_mag = mag_shifted[peak_index]
        plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
        plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq:.2f} GHz')
        plt.text(peak_freq, peak_mag + 5, f'{peak_freq:.2f} GHz', color='r', ha='center')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("FFT of QPSK After Mixing")
        plt.xlim(0, fs / 2)  # From 0 to fs in MHz
        plt.ylim(0, np.max(mag_input) + 10)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.subplot(2, 3, 5)
        peak_index = np.argmax(mag_filtered)
        peak_freq = freqs[peak_index]
        peak_mag = mag_filtered[peak_index]
        plt.plot(freqs, mag_filtered, label="Filtered QPSK", alpha=0.8)
        plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq:.2f} GHz')
        plt.text(peak_freq, peak_mag + 5, f'{peak_freq:.2f} GHz', color='r', ha='center')
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
    
    def plot_to_png(self, t, input_qpsk, qpsk_mixed, qpsk_filtered, fs):
        import matplotlib.pyplot as plt
        # Compute FFT
        x_t_lim = 3 / self.symbol_rate
        n = len(t)
        freqs = np.fft.fftfreq(n, d=1/fs)
        positive_freqs = freqs > 0
        positive_freq_values = freqs[positive_freqs]

        # FFT of original and shifted signals
        fft_input = np.fft.fft(input_qpsk)
        fft_shifted = np.fft.fft(qpsk_mixed)
        fft_filtered = np.fft.fft(qpsk_filtered)
        # Convert magnitude to dB
        mag_input = 20 * np.log10(np.abs(fft_input))
        mag_shifted = 20 * np.log10(np.abs(fft_shifted))
        mag_filtered = 20 * np.log10(np.abs(fft_filtered))
        plt.figure(figsize=(20, 6))

        # --- Time-domain plot: Original QPSK ---
        plt.subplot(1, 2, 1)
        plt.plot(t, np.real(input_qpsk))  # convert time to microseconds
        plt.title("Original QPSK Signal (Time Domain)")
        plt.xlabel("Time (μs)")
        plt.ylabel("Amplitude")
        plt.xlim(0, x_t_lim)
        plt.grid(True)

        plt.subplot(1, 2, 2)
        
        """
        positive_mags = mag_input[positive_freqs]
        positive_freq_values = freqs[positive_freqs]
        peak_index = np.argmax(positive_mags)
        peak_freq = positive_freq_values[peak_index]
        peak_mag = positive_mags[peak_index]
        """

        positive_mags = mag_input[positive_freqs]
        positive_freq_values = freqs[positive_freqs]
        peak_freq, peak_mag = find_peak(positive_mags, positive_freq_values)

        plt.plot(freqs, mag_input, label="Original QPSK", alpha=0.8)
        plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq/1e6:.1f} MHz')
        plt.text(peak_freq + 100e6, peak_mag + 6, f'{peak_freq/1e6:.1f} MHz', color='r', ha='center')
        #plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("FFT of QPSK Before Frequency Shift")
        plt.xlim(0, fs / 2)  # From 0 to fs in MHz
        plt.ylim(0, np.max(mag_input) + 10)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('original_qpsk_rp.png')

        plt.clf()

        # --- Time-domain plot: Shifted QPSK ---
        plt.subplot(1, 2, 1)
        plt.plot(t, np.real(qpsk_mixed))
        plt.title("Shifted QPSK Signal (Time Domain)")
        plt.xlabel("Time (μs)")
        plt.ylabel("Amplitude")
        plt.xlim(0, x_t_lim)
        plt.grid(True)

        

        plt.subplot(1, 2, 2)
        """
        positive_mags = mag_shifted[positive_freqs]
        positive_freq_values = freqs[positive_freqs]
        peak_index = np.argmax(positive_mags)
        peak_freq = positive_freq_values[peak_index]
        peak_mag = positive_mags[peak_index]
        """
        positive_mags = mag_shifted[positive_freqs]
        positive_freq_values = freqs[positive_freqs]

        peak_freq, peak_mag = find_peak(positive_mags, positive_freq_values)

        plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
        plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq/1e6:.1f} MHz')
        plt.text(peak_freq + 100e6, peak_mag + 6, f'{peak_freq/1e6:.1f} MHz', color='r', ha='center')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("FFT of QPSK After Frequency Shift")
        plt.xlim(0, fs / 2)  # From 0 to fs in MHz
        plt.ylim(0, np.max(mag_input) + 10)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig('shifted_qpsk_rp.png')
        plt.clf()

        plt.subplot(1, 2, 1)
        plt.plot(t, np.real(qpsk_filtered))
        plt.title("Filtered QPSK Signal (Time Domain)")
        plt.xlabel("Time (μs)")
        plt.ylabel("Amplitude")
        plt.xlim(0, x_t_lim)
        plt.grid(True)

        plt.subplot(1, 2, 2)
        
        """
        positive_mags = mag_filtered[positive_freqs]
        positive_freq_values = freqs[positive_freqs]
        peak_index = np.argmax(positive_mags)
        peak_freq = positive_freq_values[peak_index]
        peak_mag = positive_mags[peak_index]"""
        #print(freqs[peak_index-3:peak_index+3])

        positive_mags = mag_filtered[positive_freqs]
        positive_freq_values = freqs[positive_freqs]
        peak_freq, peak_mag = find_peak(positive_mags, positive_freq_values)

        plt.plot(freqs, mag_filtered, label="Filtered QPSK", alpha=0.8)
        plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq/1e6:.1f} MHz')
        plt.text(peak_freq, peak_mag + 6, f'{peak_freq/1e6:.1f} MHz', color='r', ha='center')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("FFT of QPSK After Filtering")
        plt.xlim(0, fs / 2)  # From 0 to fs in MHz
        plt.ylim(0, np.max(mag_input) + 10)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig('filtered_qpsk_rp.png')
        plt.clf()


    def handler(self, t, qpsk_waveform, f_carrier):
        
        qpsk_mixed = self.mix(qpsk_waveform, f_carrier, t)
        
        cutoff_freq = self.desired_frequency + 30e6
        
        qpsk_filtered = self.filter(cutoff_freq, qpsk_mixed)
        
        self.plot_to_png(t, qpsk_waveform, qpsk_mixed, qpsk_filtered, self.sampling_frequency)

        self.qpsk_mixed = qpsk_mixed
        self.qpsk_filtered = qpsk_filtered
