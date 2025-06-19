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


    def apply_channel(self, t):
        import numpy as np
        #generate AWGN based on noise power
        noise_standard_deviation = np.sqrt(self.noise_power / 2)
        noise_real = np.random.normal(0, noise_standard_deviation, len(self.incoming_signal))
        noise_imag = np.random.normal(0, noise_standard_deviation, len(self.incoming_signal))
        #apply single tap channel to incomiong signal
        AWGN = noise_real + 1j * noise_imag#we need real and imaginary noise
        
        #define doppler
        doppler_effect = np.exp(1j * 2 * np.pi * self.freq_shift * t)
        signal_with_doppler = doppler_effect * self.incoming_signal #apply the freq shift 
        outgoing_signal = self.h * signal_with_doppler + AWGN # element wise addition with AWGN 
        self.outgoing_signal = outgoing_signal
        return outgoing_signal
    
    def handler(self, t, Fs):
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

        plt.figure(figsize=(10, 6))
        plt.plot(t, np.real(self.incoming_signal))
        plt.title('Incoming Signal Time Domain')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(f'media/channel_{direction}_incoming_time', dpi=300)


        #plot outgoing signal in time

        plt.figure(figsize= (10, 6))
        plt.plot(t, np.real(self.outgoing_signal))
        plt.title('Incoming Signal Time Domain')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(f'media/channel_{direction}_outgoing_time', dpi=300)

        # get the fft incoming
        plt.figure(figsize = (10, 6))
        S = np.fft.fft(self.incoming_signal)
        S = np.fft.fftshift(S)
        S_mag_db = 20 * np.log10(np.abs(S))
        N = len(t)
        f = np.fft.fftshift(np.fft.fftfreq(N, d = 1/Fs))
        plt.plot(f, S_mag_db)
        plt.title('Frequency Domain of Incoming Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.tight_layout()
        plt.savefig(f'media/channel_{direction}_incoming_fft', dpi = 300)
        
        # get fft outgoing
        plt.figure(figsize = (10, 6))
        S = np.fft.fft(self.outgoing_signal)
        S = np.fft.fftshift(S)
        S_mag_db = 20 * np.log10(np.abs(S))
        f = np.fft.fftshift(np.fft.fftfreq(N, d = 1/Fs))
        plt.plot(f, S_mag_db)
        plt.title('Frequency Domain of Outgoing Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.tight_layout()
        plt.savefig(f'media/channel_{direction}_outgoing_fft', dpi = 300)


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
        plt.plot([0, np.real(h_normalized)], [0, np.imag(h_normalized)], marker='o', color='b', label='h')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.title(f'Phase of h: {np.degrees(phase):.2f}°')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'media/channel_{direction}_h_phase', dpi=300)