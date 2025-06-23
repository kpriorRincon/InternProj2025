#helper function
def fractional_delay(signal, delay_in_sec, Fs):
    """
    Apply fractional delya using interpolation (sinc-based)
    'signal' a signal to apply the time delay
    'delay_in_seconds' delay to apply to signal
    'Fs' Sample rate used to convert delay to units of samples
    """
    import numpy as np 
    
    delay_in_samples = Fs * delay_in_sec # samples/seconds * seconds = samples
    
    #filter taps
    N = 301
    #construct filter
    n = np.arange(-N//2, N//2)
    h = np.sinc(n-delay_in_samples)
    h *= np.hamming(N) #something like a rectangular window
    h /= np.sum(h)

    #apply filter
    new_signal = np.convolve(signal, h)

    #cut out Group delay from FIR filter
    delay = (N - 1) // 2 # group delay of FIR filter is always (N - 1) / 2 samples, N is filter length (of taps)
    padded_signal = np.pad(new_signal, (0, delay), mode='constant')
    new_signal = padded_signal[delay:]  # Shift back by delay

    #create a new time vector with the correcct size
    new_t = np.arange(len(new_signal)) / Fs
    
    return new_t, new_signal


class Channel: 
    def __init__(self, incoming_signal, h, noise_power, freq_shift, up = True):
        """
        Initialize a Channel object representing either an uplink or downlink channel.
        Parameters:
            incoming_signal: The input signal to the channel.
            h: Channel coefficient representing attenuation and random phase rotation (single tap).
            noise_power: The power of the noise to be added to the signal.
            freq_shift: The frequency shift applied to the signal.
            up (bool, optional): Boolean flag indicating the channel direction.
                If True, represents an uplink channel; if False, represents a downlink channel.
        """
        
        self.incoming_signal = incoming_signal
        self.outgoing_signal = None # none until it gets computed
        self.h = h #single tap channel basically attenuation and a random phase rotation
        self.noise_power = noise_power
        self.freq_shift = freq_shift
        self.up = up


    def apply_channel(self, t, time_delay):
        import numpy as np
        Fs = 1 / (t[1] - t[0]) #extract the sample rate from time vector

        # Generate complex AWGN
        noise_std = np.sqrt(self.noise_power / 2)
        AWGN = np.random.normal(0, noise_std, len(self.incoming_signal)) \
            + 1j * np.random.normal(0, noise_std, len(self.incoming_signal))

        # Apply Doppler shift
        doppler = np.exp(1j * 2 * np.pi * self.freq_shift * t)
        signal_doppler = self.incoming_signal * doppler

        # Apply single-tap channel and add noise
        signal_channel = self.h * signal_doppler
        signal_noisy = signal_channel + AWGN

        # Apply fractional delay
        new_t, delayed_signal = fractional_delay(signal_noisy, time_delay, Fs)

        self.outgoing_signal = delayed_signal
        
        return new_t, delayed_signal
    
    def handler(self, t, new_t, tune_frequency, samples_per_symbol):
        """
        Handles plotting and frequency analysis of the incoming signal.
        This method generates and saves plots of the real and imaginary parts of the incoming signal
        in the time domain, as well as its magnitude spectrum in the frequency domain.

        Parameters:
            t (numpy.ndarray): Time vector corresponding to the samples of the signal.
            Fs (float): Sampling rate of the signal in Hz.
        """
        
        #here we would like to plot
        import matplotlib.pyplot as plt
        import numpy as np 
        
        direction = 'up' if self.up else 'down'#specifier so that the user files can be differentiated 

        #plot incoming signal in time 
        #plot them side by side

        # plt.figure(figsize=(10, 6))
        # plt.plot(t, np.real(self.incoming_signal))
        # plt.title('Incoming Signal Time Domain')
        # plt.xlabel('Time')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.savefig(f'media/channel_{direction}_incoming_time', dpi=300)
        # plt.close()

        #plot outgoing signal in time

        # plt.figure(figsize= (10, 6))
        # plt.plot(t, np.real(self.outgoing_signal))
        # plt.title('Outgoing Signal Time Domain')
        # plt.xlabel('Time')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.savefig(f'media/channel_{direction}_outgoing_time', dpi=300)
        # plt.close()
        # get the fft incoming
        # plt.figure(figsize = (10, 6))
        # S = np.fft.fft(self.incoming_signal)
        # S_mag_db = 20 * np.log10(np.abs(S))
        # N = len(t)

        # f = np.fft.fftshift(np.fft.fftfreq(N, d = 1/Fs))
        # plt.plot(f, S_mag_db)
        # plt.title('Frequency Domain of Incoming Signal')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Magnitude (dB)')
        # plt.tight_layout()
        # plt.savefig(f'media/channel_{direction}_incoming_fft', dpi = 300)
        # plt.close()

        # get fft outgoing
        # plt.figure(figsize = (10, 6))
        # S = np.fft.fft(self.outgoing_signal)
        # S_mag_db = 20 * np.log10(np.abs(S))
        # f = np.fft.fftshift(np.fft.fftfreq(N, d = 1/Fs))
        # plt.plot(f, S_mag_db)
        # plt.title('Frequency Domain of Outgoing Signal')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Magnitude (dB)')
        # plt.tight_layout()
        # plt.savefig(f'media/channel_{direction}_outgoing_fft', dpi = 300)
        # plt.close()

        #plot the phase rotation h causes 
        # Normalize h to unit magnitude
        h_normalized = self.h / np.abs(self.h)
        phase = np.angle(h_normalized)
        #plotting the phase 
        plt.figure(figsize=(5, 5))
        # Plot the unit circle
        circle = plt.Circle((0, 0), 1, color='lightgray', fill=False, linestyle='--')
        plt.gca().add_artist(circle)
        # Plot the arc from 0 to phase
        arc_theta = np.linspace(0, phase, 100)
        plt.plot(np.cos(arc_theta), np.sin(arc_theta), color='orange', linewidth=2, label='Phase Arc')
        # Plot the vector for h
        plt.plot([0, np.real(h_normalized)], [0, np.imag(h_normalized)], marker='o', color='b', label='h normalized')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.title(f'Phase of h: {np.degrees(phase):.2f}°')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'media/channel_{direction}_h_phase.png', dpi=300)
        plt.close()

        tuned_signal = self.incoming_signal * np.exp(-1j * 2 * np.pi * tune_frequency * t )# we want to tune down to baseband
        #plot fft of the baseband incoming signal
        plt.figure(figsize=(10, 6))
        S = np.fft.fft(tuned_signal)
        S_mag_db = 20 * np.log10(np.abs(S))
        N = len(t)
        Fs = 1 / (t[1] - t[0])
        f = np.fft.fftshift(np.fft.fftfreq(N, d=1/Fs))
        plt.plot(f, np.fft.fftshift(S_mag_db))
        plt.title('Frequency Domain of Tuned Incoming Signal (Baseband)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.xlim(-Fs/6, Fs/6)
        plt.tight_layout()
        plt.savefig(f'media/channel_{direction}_incoming_tuned_fft.png', dpi=300)
        plt.close()

        # Plot constellation of the tuned incoming signal
        plt.figure(figsize=(6, 6))
        symbol_indices = np.arange(0, len(tuned_signal), int(samples_per_symbol))
        print(f'the symbol incidies: {symbol_indices}')
        plt.scatter(np.real(tuned_signal), np.imag(tuned_signal), color='blue', s=10, label='Oversampled')
        #this should be where the symbols actually are
        plt.scatter(np.real(tuned_signal[symbol_indices]), np.imag(tuned_signal[symbol_indices]), color='red', s=30, label='Symbol Samples')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plt.title('Constellation of Tuned Incoming Signal')
        plt.grid(True)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(f'media/channel_{direction}_incoming_tuned_constellation.png', dpi=300)
        plt.close()


        #plot fft of the baseband outgoing singla
        tuned_outgoing_signal = self.outgoing_signal * np.exp(-1j * 2 * np.pi * tune_frequency * new_t)
        N = len(new_t)
        Fs = 1 / (new_t[1] - new_t[0])
        f = np.fft.fftshift(np.fft.fftfreq(N, d=1/Fs))
        plt.figure(figsize=(10, 6))
        S_out = np.fft.fft(tuned_outgoing_signal)
        S_out_mag_db = 20 * np.log10(np.abs(S_out))
        plt.plot(f/1e6, np.fft.fftshift(S_out_mag_db))
        plt.title('Frequency Domain of Tuned Outgoing Signal (Baseband)')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magnitude (dB)')
        # plt.xlim(-Fs/6, Fs/6)
        plt.tight_layout()
        plt.savefig(f'media/channel_{direction}_outgoing_tuned_fft.png', dpi=300)
        plt.close()

        # Plot constellation of the tuned outgoing signal
        plt.figure(figsize=(6, 6))
        symbol_indices = np.arange(0, len(tuned_outgoing_signal), int(samples_per_symbol))
        plt.scatter(np.real(tuned_outgoing_signal), np.imag(tuned_outgoing_signal), color='blue', s=10, label='Oversampled')
        plt.scatter(np.real(tuned_outgoing_signal[symbol_indices]), np.imag(tuned_outgoing_signal[symbol_indices]), color='red', s=30, label='Interpreted Symbol Samples')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plt.title('Constellation of Tuned Outgoing Signal')
        plt.grid(True)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(f'media/channel_{direction}_outgoing_tuned_constellation.png', dpi=300)
        plt.close()