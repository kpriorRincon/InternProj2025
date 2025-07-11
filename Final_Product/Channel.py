from config import *
#helper function
def fractional_delay(t, signal, delay_in_sec, Fs):
    """
    Apply fractional delya using interpolation (sinc-based)
    'signal' a signal to apply the time delay
    'delay_in_seconds' delay to apply to signal
    'Fs' Sample rate used to convert delay to units of samples
    """
    import numpy as np 
    #first step we need to shift by integer 
    #for now we are only gonna get the fractional part of the delay
    total_delay = delay_in_sec * Fs # seconds * samples/second = samples
    fractional_delay = total_delay % 1 # to get the remainder
    integer_delay = int(total_delay)

    print(f'fractional delay in samples: {fractional_delay}')
    #then shift by fractional samples with the leftover 
    delay_in_samples = fractional_delay
    #Fs * delay_in_sec # samples/seconds * seconds = samples
    
    #pad with zeros for the integer_delay
    if integer_delay > 0:
        signal = np.concatenate([np.zeros(integer_delay, dtype=complex), signal])
    
    #filter taps
    N = NUMTAPS
    #construct filter
    n = np.linspace(-N//2, N//2,N)
    h = np.sinc(n-delay_in_samples)
    h *= np.hamming(N) #something like a rectangular window
    h /= np.sum(h) #normalize to get unity gain, we don't want to change the amplitude/power


    #apply filter: same time-aligned output with the same size as the input
    new_signal = np.convolve(signal, h, mode='full')
    delay = (N - 1) // 2
    new_signal = new_signal[delay:delay+len(signal)]

    #debug
    # if len(new_signal) == len(signal):
    #     print('signal lengths are the same: with mode = full and trimming')

    #create a new time vector with the correcct size
    new_t = np.arange(len(new_signal)) / Fs #longer time vector because the signal is delayed and padded in the front 
    
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
        """
        convert x(t) --> hx(t - T) +n(T) from channel effects
        """
        import numpy as np
        Fs = int (1 / (t[1] - t[0])) #extract the sample rate from time vector

        # Apply Doppler shift
        doppler = np.exp(1j * 2 * np.pi * self.freq_shift * t)
        signal_doppler = self.incoming_signal * doppler

        # Apply single-tap channel h
        signal_channel = self.h * signal_doppler
        
        # Apply fractional delay
        new_t, delayed_signal = fractional_delay(t, signal_channel, time_delay, Fs)
        
        #apply noise after delay
        AWGN = np.sqrt(self.noise_power / 2) * (np.random.randn(*delayed_signal.shape) + 1j * np.random.randn(*delayed_signal.shape))
        signal_noisy = delayed_signal + AWGN

        self.outgoing_signal = signal_noisy
        
        return new_t, signal_noisy
    
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
        plt.rcParams.update({
            'axes.titlesize': 20,
            'axes.labelsize': 16,      # X and Y axis label size
            'xtick.labelsize': 14,     # X tick label size
            'ytick.labelsize': 14,     # Y tick label size
            'legend.fontsize': 14,     # Legend font size
            'figure.titlesize': 22     # Figure suptitle size
        })
        import numpy as np 
        
        direction = 'up' if self.up else 'down'#specifier so that the user files can be differentiated 

    
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
        
        plt.savefig(f'media/channel_{direction}_h_phase.png', dpi=300)
        plt.close()

        tuned_signal = self.incoming_signal * np.exp(-1j * 2 * np.pi * tune_frequency * t)# we want to tune down to baseband

        # Plot constellation of the tuned incoming signal
        plt.figure(figsize=(6, 6))
        symbol_indices = np.arange(0, len(tuned_signal), int(samples_per_symbol))
        # print(f'the symbol incidies: {symbol_indices}')
        plt.plot(np.real(tuned_signal), np.imag(tuned_signal), 'b-', label='Oversampled', zorder = 1)
        #this should be where the symbols actually are
        
        plt.scatter(np.real(tuned_signal[symbol_indices]), np.imag(tuned_signal[symbol_indices]), color='red', s=30, label='Symbol Samples', zorder = 2)
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plt.title('Constellation of Tuned\n Incoming Signal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.savefig(f'media/channel_{direction}_incoming_tuned_constellation.png', dpi=300)
        plt.close()


        #plot fft of the baseband outgoing signal
        tuned_outgoing_signal = self.outgoing_signal * np.exp(-1j * 2 * np.pi * tune_frequency * new_t)
        
        
        

        # Plot constellation of the tuned outgoing signal
        plt.figure(figsize=(6, 6))
        symbol_indices = np.arange(0, len(tuned_outgoing_signal), int(samples_per_symbol))
        plt.plot(np.real(tuned_outgoing_signal), np.imag(tuned_outgoing_signal), 'b-',zorder = 1, label='Oversampled')
        plt.scatter(np.real(tuned_outgoing_signal[symbol_indices]), np.imag(tuned_outgoing_signal[symbol_indices]), color='red', s=30, label='Interpreted Symbol Samples', zorder =2)
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plt.title('Constellation of Tuned\n Outgoing Signal')
        plt.grid(True)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.savefig(f'media/channel_{direction}_outgoing_tuned_constellation.png', dpi=300)
        plt.close()