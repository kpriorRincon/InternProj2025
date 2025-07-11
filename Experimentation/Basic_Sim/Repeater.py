import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def find_peak(signal, sample_rate, top_n_bins=5):
    N = len(signal)
    spectrum = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(N, d=1/sample_rate)

    if np.isrealobj(signal):
        half_N = N // 2
        spectrum = spectrum[:half_N]
        freqs = freqs[:half_N]

    # Get indices of the top N peaks
    top_indices = np.argsort(spectrum)[-top_n_bins:]
    
    # Calculate weighted centroid of these bins
    weights = spectrum[top_indices]
    weighted_freqs = freqs[top_indices]
    carrier_freq = np.sum(weighted_freqs * weights) / np.sum(weights)

    return carrier_freq

def attenuator(R, fc, sig):
    lam = 3e8/fc # wavelength of the signal
    fspl = (lam/(4*np.pi*R))**2
    Pt = np.max(np.abs(sig))
    Gt = 1.5
    Gr = Gt
    Pr = Pt*Gt*Gr*fspl
    sig = (sig - sig.max()) / (sig.max() - sig.min())
    return sig*Pr

def variable_amplifier(sig):
    P_target = 1
    Pr = np.max(np.abs(sig))
    gain = P_target / Pr
    return gain, gain*sig

class Repeater:
    def __init__(self, sampling_frequency, symbol_rate):
        self.desired_frequency = None  # Default frequency set to 1 GHz
        self.sampling_frequency = sampling_frequency
        self.symbol_rate = symbol_rate
        self.gain = None
        self.R = 2000e3 # typical leo height

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
        # don't apply filter

        # Implement filtering logic here

        numtaps = 101  # order of filter
        lowcut = self.desired_frequency - 50e6 #850e6
        highcut = self.desired_frequency + 50e6 #960e6
        fir_coeff = signal.firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=self.sampling_frequency)
        # pass-zero = whether DC / 0Hz is in the passband
        
        filtered_sig = signal.lfilter(fir_coeff, 1.0, mixed_qpsk)
        #first param is for coefficients in numerator (feedforward) of transfer function
        #sec param is for coeff in denom (feedback)
        #FIR are purely feedforward, as they do not depend on previous outputs

        delay = (numtaps - 1) // 2 # group delay of FIR filter is always (N - 1) / 2 samples, N is filter length (of taps)
        padded_signal = np.pad(filtered_sig, (0, delay), mode='constant')
        filtered_sig = padded_signal[delay:]  # Shift back by delay

        #b, a = signal.butter(order, cuttoff_frequency, btype='low', fs=self.sampling_frequency) # butterworth filter coefficients

        # Apply filter
        #filtered_sig = signal.filtfilt(b, a, mixed_qpsk)   # filtered signal
        
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
        plt.ylim(-50, np.max(mag_input) + 10)
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
        window = np.hanning(n)

        freqs = np.fft.fftfreq(n, d=1/fs)
        positive_freqs = freqs > 0
        positive_freq_values = freqs[positive_freqs]

        # FFT of original and shifted signals
        fft_input = np.fft.fft(input_qpsk * window)
        fft_shifted = np.fft.fft(qpsk_mixed * window)
        fft_filtered = np.fft.fft(qpsk_filtered * window) 
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

        peak_mag = 50
        positive_mags = mag_input[positive_freqs]
        positive_freq_values = freqs[positive_freqs]
        peak_freq = find_peak(input_qpsk, self.sampling_frequency)

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

        peak_freq = find_peak(qpsk_mixed, self.sampling_frequency)

        plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
        plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq/1e6:.1f} MHz')
        plt.text(peak_freq + 100e6, peak_mag + 6, f'qq{peak_freq/1e6:.1f} MHz', color='r', ha='center')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("FFT of QPSK After FrequeXXncy Shift")
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
        peak_freq = find_peak(qpsk_mixed, self.sampling_frequency)

        plt.plot(freqs, mag_filtered, label="Filtered QPSK", alpha=0.8)
        plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq/1e6:.1f} MHz')
        plt.text(peak_freq, peak_mag + 6, f'{peak_freq/1e6:.1f} MHz', color='r', ha='center')
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("FFT of QPSK After Filtering")
        plt.xlim(0, fs / 2)  # From 0 to fs in MHz
        plt.ylim(20, np.max(mag_input) + 10)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig('filtered_qpsk_rp.png')
        plt.clf()


    def handler(self, t, qpsk_waveform, f_carrier):
        attenuated_signal = attenuator(self.R, f_carrier, qpsk_waveform)            
        calculated_gain, amplified_signal = variable_amplifier(attenuated_signal)   # TODO implement in rest of code, i don't want to touch it bc of gui stuff -Trevor
        qpsk_mixed = self.mix(qpsk_waveform, f_carrier, t)        
        cuttoff_freq = self.desired_frequency + 30e6
        qpsk_filtered = self.filter(cuttoff_freq, qpsk_mixed)
        qpsk_filtered = self.amplify(qpsk_filtered)
        self.plot_to_png(t, qpsk_waveform, qpsk_mixed, qpsk_filtered, self.sampling_frequency)

        self.qpsk_mixed = qpsk_mixed
        self.qpsk_filtered = qpsk_filtered

    
