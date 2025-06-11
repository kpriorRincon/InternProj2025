class SigGen:
    
    def __init__(self, freq=1.0, amp=1.0, sample_rate =100000, symbol_rate = 1000):
        import numpy as np
        self.freq = freq  # Frequency in Hz
        self.sample_rate = sample_rate  # sample rate in samples per second
        self.symbol_rate = symbol_rate  # Symbol rate in symbols per second
        self.amp = amp    # Amplitude

        # Map bit pairs to complex symbols
        self.mapping = {
            (0, 0): (1 + 1j) / np.sqrt(2),
            (0, 1): (-1 + 1j) / np.sqrt(2),
            (1, 1): (-1 - 1j) / np.sqrt(2),
            (1, 0): (1 - 1j) / np.sqrt(2)
        }

    # Add noise to symbols
    def noise_adder(self, sinusoid, noise_power):
        import numpy as np
        # Noise parameters
        mean_noise = 0                                                  # Mean of the noise distribution
        std_noise = noise_power                                                 # Standard deviation of the noise distribution

        # Generate noise
        noise_real = np.random.normal(mean_noise, std_noise/np.sqrt(2), len(sinusoid))
        noise_imag = np.random.normal(mean_noise, std_noise/np.sqrt(2), len(sinusoid))
        noise = noise_real + 1j*noise_imag
        return sinusoid + noise                                         # returns the sinusoid with added noise
    
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

    def generate_qpsk(self, bits, bool_noise, noise_power = 0.01):
        """
        Generate a QPSK signal from a sequence of bits.
        
        Parameters:
            bits (list): List of bits (0s and 1s) len(bits) % 2 == 0.
            sample_rate (int): Number of samples per second.
            symbol_rate (int): Number of symbols per second.
        
        Returns:
            np.ndarray: Time vector.
            np.ndarray: QPSK waveform.
            list: Vertical lines to show phase transition.
            symbols (list): List of complex symbols corresponding to the bit pairs.
        """
        import numpy as np
        
        # Convert bits to symbols
        if len(bits) % 2 != 0:
            raise ValueError("Bit sequence must have an even length.")
        
        # Map bit pairs to complex symbols
        symbols = [self.mapping[(bits[i], bits[i + 1])] for i in range(0, len(bits), 2)]
        
        # Calculate samples per symbol
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        
        # Create time vector for the entire waveform
        total_samples = len(symbols) * samples_per_symbol
        #create a time vector that is total symbols long
        t = np.arange(total_samples) / self.sample_rate
        
        # Upsample symbols to match sampling rate
        # Each symbol is held constant for samples_per_symbol duration
        upsampled_symbols = np.concatenate([np.append(x, np.zeros(samples_per_symbol-1, dtype=complex))for x in symbols])

        # Root raised cosine filter implementation
        from commpy import filters
        beta = 0.3
        _, pulse_shape = self.rrc_filter(beta, 300, 1/self.symbol_rate, self.sample_rate)
        # pulse_shape = np.convolve(pulse_shape, pulse_shape)/2
        signal = np.convolve(pulse_shape, upsampled_symbols, 'same')

        # Generate complex phasor at carrier frequency
        phasor = np.exp(1j * 2 * np.pi * self.freq * t)

        
        # Modulate: multiply upsampled symbols by phasor
        qpsk_waveform = signal * phasor

        # get vertical lines
        t_vertical_lines = []  # Initialize vertical lines for debugging
        for i, symbol in enumerate(symbols):
            #compute the phase offset for the symbol
            phase_offset = np.angle(symbol)
            #debugging print statement to show the symbol and phase offset
            #print(f"Symbol {i}: {symbol}, Phase Offset: {phase_offset}");
            idx_start = i * samples_per_symbol
            t_vertical_lines.append(idx_start/self.sample_rate)

        if bool_noise:
            # add noise to the QPSK wavefrorm
            qpsk_waveform = self.noise_adder(qpsk_waveform, noise_power)

        return t, qpsk_waveform, symbols, t_vertical_lines, upsampled_symbols
    
    def message_to_bits(self, message):
        """
        Author: Skylar Harris
        Convert a string message to a list of bits.
        
        Parameters:
            message (str): The input string message.
        
        Returns:
            list: List of bits (0s and 1s).
        """
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
        
        message_binary = ''.join(format(ord(x), '08b') for x in message)

        # Add start and end sequences to the message binary
        message_binary = ''.join(str(bit) for bit in start_sequence) + message_binary + ''.join(str(bit) for bit in end_sequence)
        print(message_binary)
        # Convert string input to list of integers
        bit_sequence = [int(bit) for bit in message_binary.strip()]
        return bit_sequence
    
