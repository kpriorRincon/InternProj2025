# TODO we need to build this class out from the file:
#sim_qpsk_noisy_demod.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import fftconvolve
import Sig_Gen_Noise as SigGen


def attenuator(R, fc, sig):
    lam = 3e8/fc # wavelength of the signal
    fspl = (lam/(4*np.pi*R))**2
    Pt = np.max(np.abs(sig))
    Gt = 1.5
    Gr = Gt
    Pr = Pt*Gt*Gr*fspl
    sig = (sig - sig.max()) / (sig.max() - sig.min())
    return sig*Pr

class Receiver:
    def __init__(self, sampling_rate):
        #constructor
        self.sampling_rate = sampling_rate
        self.frequency = None
        # Phase sequence for qpsk modulation corresponds to the letter 'R'
        self.phase_start_sequence = np.array([-1+1j, -1+1j, 1+1j, 1-1j]) # this is the letter R in QPSK
        #self.phases = np.array([45, 135, 225, 315])  # QPSK phase angles in degrees
        self.phase_start_sequence = np.array([-1-1j, -1-1j, 1-1j, -1+1j, 
                                 1-1j, 1-1j, -1+1j, 1+1j,
                                 1+1j, 1-1j, 1-1j, -1-1j,
                                 1-1j, -1-1j, 1+1j, -1+1j])    # gold code start 
                                                                # 11 11 10 01 
                                                                # 10 10 01 00 
                                                                # 00 10 10 11 
                                                                # 10 11 00 01
        self.phase_end_sequence = np.array([1+1j, 1-1j, -1+1j, 1-1j,
                                    1-1j, 1+1j, 1+1j, 1-1j,
                                    1+1j, -1-1j, -1-1j, -1+1j,
                                    1+1j, -1+1j, 1+1j, 1-1j])   # gold code end 
                                                                    # 00 10 01 10 
                                                                    # 10 00 00 10 
                                                                    # 00 11 11 01 
                                                                    # 00 01 00 10
        self.R = 2000e3 # typical leo height
        self.attenuate = False

    # Sample the received signal
    def down_sampler(self,sig, sample_rate, symbol_rate):
        # write some downsampling here
        samples_per_symbol = int(sample_rate/symbol_rate)
        symbols = sig[::samples_per_symbol]
        return symbols
    
    # QPSK symbol to bit mapping
    def bit_reader(self, symbols):
        # print("Reading bits from symbols")
        bits = np.zeros((len(symbols), 2), dtype=int)
        for i in range(len(symbols)):
            angle = np.angle(symbols[i], deg=True) % 360

            # codex for the phases to bits
            if 0 <= angle < 90:
                bits[i] = [0, 0]  # 45°
            elif 90 <= angle < 180:
                bits[i] = [0, 1]  # 135°
            elif 180 <= angle < 270:
                bits[i] = [1, 1]  # 225°
            else:
                bits[i] = [1, 0]  # 315°
        return bits

    # Error checking for the start sequence
    def error_handling(self, sampled_symbols):
        #print("Error checking")
        #print("Error checking")
        ## look for the start sequence ##
        expected_start_sequence = ''.join(str(bit) for pair in self.bit_reader(self.phase_start_sequence) for bit in pair)  # put the start sequence into a string
        best_bits = None                                                                                                    # holds the best bits found
        #print("Expected Start Sequence: ", expected_start_sequence)                                                         # debug statement
        #print("Expected Start Sequence: ", expected_start_sequence)                                                         # debug statement
        og_sampled_symbols = ''.join(str(bit) for pair in self.bit_reader(sampled_symbols) for bit in pair)                 # original sampled symbols in string format
        #print("Sampled bits: ", og_sampled_symbols)                                                                         # debug statement
        #print("Sampled bits: ", og_sampled_symbols)                                                                         # debug statement

        ## Loop through possible phase shifts ##
        for i in range(0, 3):   # one for each quadrant (0°, 90°, 180°, 270°)
            # Rotate the flat bits to match the start sequence
            rotated_bits = sampled_symbols * np.exp(-1j* np.deg2rad(i*90))  # Rotate by 0, 90, 180, or 270 degrees
            
            # decode the bits
            decode_bits = self.bit_reader(rotated_bits)                             # decode the rotated bits
            flat_bits = ''.join(str(bit) for pair in decode_bits for bit in pair)   # put the bits into a string
            #print("Rotated bits: ", flat_bits)                                      # debug statement
            #print("Rotated bits: ", flat_bits)                                      # debug statement
            
            # Check for presence of the known start sequence (first few symbols)
            if expected_start_sequence == flat_bits[0:8]:                   # check only first 8 symbols worth (16 bits)
                #print(f"Start sequence found with phase shift: {i*90}°")
                #print(f"Start sequence found with phase shift: {i*90}°")
                best_bits = flat_bits                                       # store the best bits found
                break
        
        # Error state if no start sequence was found
        if best_bits is None:
            #print("Start sequence not found. Defaulting to 0°")
            #print("Start sequence not found. Defaulting to 0°")
            rotated_symbols = sampled_symbols
            decoded_bits = self.bit_reader(rotated_symbols)
            best_bits = ''.join(str(b) for pair in decoded_bits for b in pair)
        
        return best_bits
    
    def cross_correlation(self, baseband_sig, sample_rate, symbol_rate):
        start_index = 0
        end_index = len(baseband_sig)
        
        # Define start and end sequences
        # 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 0 0 0 1 0 1 0 1 1 1 0 1 1 0 0 0 1
        start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                        1, 0, 1, 0, 0, 1, 0, 0,
                        0, 0, 1, 0, 1, 0, 1, 1,
                        1, 0, 1, 1, 0, 0, 0, 1]
        
        # 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0 1 0 0 1 0
        end_sequence = [0, 0, 1, 0, 0, 1, 1, 0,
                        1, 0, 0, 0, 0, 0, 1, 0, 
                        0, 0, 1, 1, 1, 1, 0, 1, 
                        0, 0, 0, 1, 0, 0, 1, 0]
        
        #print(f"Looking for start sequence: {start_sequence}")
        #print(f"Looking for end sequence: {end_sequence}")

        sig_gen = SigGen.SigGen(0, 1.0, sample_rate, symbol_rate)
        _, start_waveform, _, _,_= sig_gen.generate_qpsk(start_sequence, False, 0.1)
        _, end_waveform, _, _, _= sig_gen.generate_qpsk(end_sequence, False, 0.1)
        
        # Correlate with start sequence
        correlated_signal = fftconvolve(baseband_sig, np.conj(np.flip(start_waveform)), mode='full')
        end_cor_signal = fftconvolve(baseband_sig, np.conj(np.flip(end_waveform)), mode='full')
        
        # Find maximum correlation
        start_index = np.argmax(np.abs(correlated_signal)) - 16*int(sample_rate/symbol_rate) # go back 16 symbols e.g. 32 bits
        end_index = np.argmax(np.abs(end_cor_signal))
        
        return start_index, end_index

    def filter(self, input_signal, sample_rate):
        """
        Filters the mixed signal to remove unwanted frequencies.

        Returns:
        - The filtered signal.
        """

        # Implement filtering logic here

        numtaps = 101  # order of filter
        cutoff_freq = 500
        fir_coeff = signal.firwin(numtaps, cutoff_freq, pass_zero='lowpass', fs=sample_rate)
        
        filtered_sig = signal.lfilter(fir_coeff, 1.0, input_signal)
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
    
    def rrc_filter(self, beta, N, Ts, fs):
        
        """
        Generate a Root Raised-Cosine (RRC) filter (FIR) impulse response

        Parameters:
        - beta : Roll-off factor (0 < beta <= 1)
        - N : Total number of taps in the filter (the filter span)
        - Ts : Symbol period 
        - fs : Sampling frequency/rate (Hz)

        Returns:
        - h : The impulse response of the RRC filter in the time domain
        - time : The time vector of the impulse response

        """

        # Importing necessary libraries 
        import numpy as np
        from scipy.fft import fft, ifft

        # The number of samples in each symbol
        samples_per_symbol = int(fs * Ts)

        # The filter span in symbols
        total_symbols = N / samples_per_symbol

        # The total amount of time that the filter spans
        total_time = total_symbols * Ts

        # The time vector to compute the impulse response
        time = np.linspace(-total_time / 2, total_time / 2, N, endpoint=False)

        # ---------------------------- Generating the RRC impulse respose ----------------------------

        # The root raised-cosine impulse response is generated from taking the square root of the raised-cosine impulse response in the frequency domain

        # Raised-cosine filter impulse response in the time domain
        num = np.cos( (np.pi * beta * time) / (Ts) )
        denom = 1 - ( (2 * beta * time) / (Ts) ) ** 2
        g = np.sinc(time / Ts) * (num / denom)

        # Raised-cosine filter impulse response in the frequency domain
        fg = fft(g)

        # Root raised-cosine filter impulse response in the frequency domain
        fh = np.sqrt(fg)

        # Root raised-cosine filter impulse respone in the time domain
        h = ifft(fh)

        return time, h 
    
    # sample the received signal and do error checking
    def demodulator(self, qpsk_waveform, sample_rate, symbol_rate, t, fc):
        # attenuate signa
        if self.attenuate:
            attenuated_signal = attenuator(self.R, fc, qpsk_waveform)
            ## tune to baseband ##
            #print("Tuning to basband...")
            baseband_sig = attenuated_signal * np.exp(-1j * 2 * np.pi * fc * t)
        else:
            ## tune to baseband ##
            #print("Tuning to basband...")
            baseband_sig = qpsk_waveform * np.exp(-1j * 2 * np.pi * fc * t)
        
        # low pass filter
        filtered_sig = self.filter(baseband_sig, sample_rate)

        # root raised cosine matched filter
        beta = 0.3
        #_, pulse_shape = filters.rrcosfilter(300, beta, 1/symbol_rate, sample_rate)
        _, pulse_shape = self.rrc_filter(beta, 300, 1/symbol_rate, sample_rate)
        # pulse_shape = np.convolve(pulse_shape, pulse_shape)/2
        signal = np.convolve(pulse_shape, filtered_sig, 'same')

        #find the desired signal
        lam = 3e8 / fc  # wavelength of the carrier frequency
        #v = 7.8e3 # average speed of a satellite in LEO
        v = 0
        doppler = v / lam   # calculated doppler shift
        #print("Doppler shift: ", doppler)
        freqs = np.linspace(fc-doppler, fc+doppler, 4)
        start_index, end_index = self.cross_correlation(signal, sample_rate, symbol_rate)
        analytic_sig = signal[start_index:end_index]

        # sample the analytic signal
        #print("Sampling the analytic signal...")
        sampled_symbols = self.down_sampler(analytic_sig, sample_rate, symbol_rate)

        # decode the symbols and error check the start sequence
        #print("Decoding symbols and checking for start sequence...")
        best_bits = self.error_handling(sampled_symbols)

        return signal, sampled_symbols, best_bits, filtered_sig


    def plot_data(self, analytical_output, sampled_symbols, t):
        # constellation plot
        plt.figure(figsize=(10, 4))
        plt.scatter(np.real(sampled_symbols), np.imag(sampled_symbols))
        plt.grid(True)
        plt.title('Constellation Plot of Sampled Symbols')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.savefig('demod_media/Constellation.png')

        # Plot the waveform and phase
        plt.figure(figsize=(10, 4))
        plt.plot(t, np.real(analytical_output), label='I (real part)')
        plt.plot(t, np.imag(analytical_output), label='Q (imag part)')
        plt.title('Filtered Baseband Time Signal  (Real and Imag Parts)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.legend()
        plt.savefig('demod_media/Base_Band_Waveform.png')

        # plot the fft
        ao_fft = np.fft.fft(analytical_output)
        freqs = np.fft.fftfreq(len(analytical_output), d=1/2*self.sampling_rate)
        plt.figure(figsize=(10, 4))
        plt.plot(freqs, 20*np.log10(ao_fft))
        plt.title('FFT of the Filtered Base Band Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Madgnitude (dB)')
        plt.grid()
        plt.savefig('demod_media/Base_Band_FFT.png')

    def get_string(self, bits):
        """Convert bits to string."""
        # bits is an array
        #exclude the prefix
        bits = bits[32:-32]
        # convert the bits into a string
        return ''.join(chr(int(bits[i*8:i*8+8],2)) for i in range(len(bits)//8))

    def handler(self, qpsk_waveform, sample_rate, symbol_rate, fc, t):
        analytical_signal, best_bits = self.demodulator(qpsk_waveform, sample_rate, symbol_rate, fc) 
        self.plot_data(analytical_signal, t)
        #return these things to be displayed
        return best_bits, self.get_string(best_bits)

   