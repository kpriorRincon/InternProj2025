class Repeater:
    def __init__(self, sampling_frequency):
        self.desired_freqeuncy = None  # Default frequency set to 1 GHz
        self.sampling_fequency = sampling_frequency
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
        import numpy as np
        # Implement mixing logic here
        #Complex sinusoid (real-world)
        #mixing_signal = np.cos(2 * np.pi * (self.desired_freqeuncy + qpsk_frequency) * t)

        #Complex exponential (Ideal)
        mixing_signal = np.exp(1j * 2 * np.pi * (self.desired_freqeuncy - qpsk_frequency) * t)
        
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
        from scipy import signal

        b, a = signal.butter(order, cuttoff_frequency, btype='low', fs=self.sampling_fequency) # butterworth filter coefficients

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
        import numpy as np
        import matplotlib.pyplot as plt
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
    
    def plot_to_png(self, t, input_qpsk, qpsk_mixed, qpsk_filtered, fs):
        import numpy as np
        import matplotlib.pyplot as plt
        # Compute FFT
        n = len(t)
        freqs = np.fft.fftfreq(n, d=1/fs)

        # FFT of original and shifted signals
        fft_input = np.fft.fft(input_qpsk)
        fft_shifted = np.fft.fft(qpsk_mixed)
        fft_filtered = np.fft.fft(qpsk_filtered)
        # Convert magnitude to dB
        mag_input = 20 * np.log10(np.abs(fft_input))
        mag_shifted = 20 * np.log10(np.abs(fft_shifted))
        mag_filtered = 20 * np.log10(np.abs(fft_filtered))
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
        plt.plot(t, np.real(qpsk_mixed))
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

        plt.savefig('repeater.png')

    def handler(self, t, qpsk_waveform, f_carrier):
        
        qpsk_mixed = self.mix(qpsk_waveform, f_carrier, t)
        
        cutoff_freq = self.desired_freqeuncy + 30e6
        
        qpsk_filtered = self.filter(cutoff_freq, qpsk_mixed)
        
        self.plot_to_png(t, qpsk_waveform, qpsk_mixed, qpsk_filtered, self.sampling_fequency)

        self.qpsk_mixed = qpsk_mixed
        self.qpsk_filtered = qpsk_filtered