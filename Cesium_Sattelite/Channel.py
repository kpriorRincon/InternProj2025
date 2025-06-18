class Channel: 
    def __init__(self, h, noise_power):

        self.h = h
        self.noise_power = noise_power

    def apply_channel(self, incoming_signal):
        import numpy as np
        #apply single tap channel to incomiong signal
        #generate AWGN based on noise power
        noise_standard_deviation = np.sqrt(self.noise_power/2)
        noise_real = np.random.normal(0, noise_standard_deviation, len(incoming_signal))
        noise_imag = np.random.normal(0, noise_standard_deviation, len(incoming_signal))
        AWGN = noise_real + 1j * noise_imag#we need real and imaginary noise
        self.h * incoming_signal + AWGN # element wise addition of AWGN 

    def handler(self):
        #here we would like to plot