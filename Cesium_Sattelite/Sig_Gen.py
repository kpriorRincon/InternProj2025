class SigGen:

    def __init__(self, freq=1.0, amp=1.0, sample_rate=40e6, symbol_rate=4e6):
        import numpy as np
        self.freq = freq  # Frequency in Hz
        self.sample_rate = sample_rate  # sample rate in samples per second
        self.symbol_rate = symbol_rate  # Symbol rate in symbols per second
        self.amp = amp    # Amplitude
        self.upsampled_symbols = None
        self.pulse_shaped_symbols = None
        self.qpsk_signal = None
        # Map bit pairs to complex symbols
        self.mapping = {
            (0, 0): (1 + 1j) / np.sqrt(2),
            (0, 1): (-1 + 1j) / np.sqrt(2),
            (1, 1): (-1 - 1j) / np.sqrt(2),
            (1, 0): (1 - 1j) / np.sqrt(2)
        }

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
        num = np.cos((np.pi * beta * time) / (Ts))
        denom = 1 - ((2 * beta * time) / (Ts)) ** 2
        g = np.sinc(time / Ts) * (num / denom)

        # Raised-cosine filter impulse response in the frequency domain
        fg = fft(g)

        # Root raised-cosine filter impulse response in the frequency domain
        fh = np.sqrt(fg)

        # Root raised-cosine filter impulse respone in the time domain
        h = ifft(fh)
        h /= np.sum(h) # normalize to get unity gain, we dpon't want to change the amplitude/power
        return time, h

    def generate_qpsk(self, bits):
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
        symbols = [self.mapping[(bits[i], bits[i + 1])]
                   for i in range(0, len(bits), 2)]

        # Calculate samples per symbol
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)

        # Create time vector for the entire waveform
        total_samples = len(symbols) * samples_per_symbol
        # create a time vector that is total symbols long
        t = np.arange(total_samples) / self.sample_rate
        # Upsample symbols to match sampling rate
        #this will make an array like [(1+1j)/root2, 0, 0, 0, 0, 0, 0, 0,..., (1-1j)/root2, ]
        upsampled_symbols = np.concatenate([np.append(x, np.zeros(samples_per_symbol-1, dtype=complex))for x in symbols])
        self.upsampled_symbols = upsampled_symbols
        # Root raised cosine filter implementation
        beta = 0.3
        _, pulse_shape = self.rrc_filter(beta, 300, 1/self.symbol_rate, self.sample_rate)
        
        # pulse_shape = np.convolve(pulse_shape, pulse_shape)/2
        signal = np.convolve(pulse_shape, upsampled_symbols, 'same')
        self.pulse_shaped_symbols = signal
        # Generate complex phasor at carrier frequency
        phasor = np.exp(1j * 2 * np.pi * self.freq * t)
        # Modulate: multiply pulse shaped upsampled symbols with the complex carrier
        qpsk_waveform = signal * phasor * self.amp
        self.qpsk_signal = qpsk_waveform
        return t, qpsk_waveform

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
        message_binary = ''.join(str(bit) for bit in start_sequence) + \
            message_binary + ''.join(str(bit) for bit in end_sequence)
        # print(message_binary)
        # Convert string input to list of integers
        bit_sequence = [int(bit) for bit in message_binary.strip()]
        return bit_sequence

    def handler(self, t):
        import matplotlib.pyplot as plt
        import numpy as np
        #we would like to plot

        #upsampled_bits real and imaginary
        plt.figure(figsize = (10,6))
        plt.subplot(2, 1, 1)
        plt.plot(t, np.real(self.upsampled_symbols), 'b.-', label = 'Real')
        plt.legend()
        plt.title('Upsampled Bits I (Real Part)')
        plt.subplot(2,1,2)
        plt.plot(t, np.imag(self.upsampled_symbols), 'r.-', label = 'Imaginary')
        plt.legend()
        plt.title('Upsampled Bits Q (Imaginary Part)')
        plt.tight_layout()
        plt.savefig('media/tx_upsampled_bits.png', dpi=300)
        
        #Pulse Shaped bits
        plt.figure(figsize = (10,6))
        plt.subplot(2, 1, 1)
        plt.plot(t, np.real(self.pulse_shaped_symbols), 'b.-', label = 'Real')
        plt.legend()
        plt.title('Pulse Shaped I (Real Part)')
        plt.subplot(2,1,2)
        plt.plot(t, np.imag(self.pulse_shaped_symbols), 'r.-', label = 'Imaginary')
        plt.legend()
        plt.title('Pulse Shaped Q (Imaginary Part)')
        plt.tight_layout()
        plt.savefig('media/tx_pulse_shaped_bits.png', dpi=300)
        #plot the modulated signal 
        plt.figure(figsize=(10, 6))
        plt.plot(t, np.real(self.qpsk_signal))
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Snippet of Modulated Signal")
        plt.tight_layout()
        plt.savefig('media/tx_waveform_snippet.png', dpi=300)

        #


        #