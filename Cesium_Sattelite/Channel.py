class Channel: 
    def __init__(self, incoming_signal, h, noise_power, freq_shift):
        
        self.incoming_signal = incoming_signal
        self.h = h #single tap channel basically attenuation and a random phase rotation
        self.noise_power = noise_power
        self.freq_shift = freq_shift

    def apply_channel(self, t):
        import numpy as np
        #generate AWGN based on noise power
        noise_standard_deviation = np.sqrt(self.noise_power/2)
        noise_real = np.random.normal(0, noise_standard_deviation, len(self.incoming_signal))
        noise_imag = np.random.normal(0, noise_standard_deviation, len(self.incoming_signal))
        #apply single tap channel to incomiong signal
        AWGN = noise_real + 1j * noise_imag#we need real and imaginary noise
        
        #define doppler
        doppler_effect = np.exp(1j * 2 * np.pi * self.freq_shift * t)
        signal_with_doppler = doppler_effect * self.incoming_signal #apply the freq shift 
        outgoing_signal = self.h * signal_with_doppler + AWGN # element wise addition with AWGN 
        return outgoing_signal
    
    def handler(self, t):
        #here we would like to plot
        import matplotlib.pyplot as plt
        pass # for now