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
        - time : The time vector of the impulse response
        - h : The impulse response of the RRC filter in the time domain
        """

        import numpy as np 

        # Time vector centered at zero
        t = np.arange(-N // 2, N // 2 + 1) / fs #symmetric and centered N must be odd

        h = np.zeros_like(t) # start with zeros for the length of the time vector

        for i in range(len(t)):
            #populate h based on the impusle response
            if t[i] == 0.0:
                h[i] = (1.0 + beta * (4/np.pi - 1))
            elif abs(t[i]) == Ts / (4 * beta):
                h[i] = (beta / np.sqrt(2)) * (
                    ((1 + 2/np.pi) * np.sin(np.pi / (4 * beta))) +
                    ((1 - 2/np.pi) * np.cos(np.pi / (4 * beta)))
                )
            else:
                numerator = np.sin(np.pi * t[i] * (1 - beta) / Ts) + 4 * beta * t[i] / Ts * np.cos(np.pi * t[i] * (1 + beta) / Ts)
                denominator = np.pi * t[i] * (1 - (4 * beta * t[i] / Ts) ** 2) / Ts
                h[i] = numerator / denominator
        #debug
        # import matplotlib.pyplot as plt
        # plt.plot(t, h, '.-')
        # #plt.plot(t, np.convolve(h, h, mode='same'), '.-')
        # plt.show()
        return t, h 

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
        """
        import numpy as np
        from scipy.signal import resample_poly
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
        upsampled_symbols = np.zeros(len(symbols)*samples_per_symbol, dtype = complex)
        upsampled_symbols[::samples_per_symbol] = symbols
        self.upsampled_symbols = upsampled_symbols
        
        # Root raised cosine filter implementation
        beta = 0.4
        _, pulse_shape = self.rrc_filter(beta, 301, 1/self.symbol_rate, self.sample_rate)

        signal = np.convolve(upsampled_symbols, pulse_shape, mode='same')

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

        #upsampled_bits real and imaginary with dashed stems at symbol locations
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        symbol_indices = np.arange(0, len(self.upsampled_symbols), samples_per_symbol)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, np.real(self.upsampled_symbols), 'b.-', label='Real')
        plt.legend() 
        plt.title('Upsampled Bits I (Real Part)')

        plt.subplot(2, 1, 2)
        plt.plot(t, np.imag(self.upsampled_symbols), 'r.-', label='Imaginary')
        plt.legend()
        plt.title('Upsampled Bits Q (Imaginary Part)')

        plt.tight_layout()
        plt.savefig('media/tx_upsampled_bits.png', dpi=300)
        plt.close()

        # Compute FFT of the upsampled bits (before pulse shaping)
        upsampled = self.upsampled_symbols
        N_upsampled = len(upsampled)
        fft_upsampled = np.fft.fftshift(np.fft.fft(upsampled))
        freqs_upsampled = np.fft.fftshift(np.fft.fftfreq(N_upsampled, d=1/self.sample_rate))

        plt.figure(figsize=(10, 6))
        plt.plot(freqs_upsampled / 1e6, 20 * np.log10(np.abs(fft_upsampled)))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("FFT of Upsampled Bits (Baseband)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('media/tx_upsampled_bits_fft.png', dpi=300)
        plt.close()
        
        #pulse shaping impulse response:
        plt.figure(figsize=(5,5))
        pulse_t,pulse_shape = self.rrc_filter(0.4, 301, 1/self.symbol_rate, self.sample_rate)
        plt.plot(pulse_t, pulse_shape, 'o')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("RRC Filter Impulse Response")
        plt.tight_layout()
        plt.savefig('media/tx_rrc.png', dpi = 300)
        #Pulse Shaped bits
        plt.figure(figsize = (10,6))
        plt.subplot(2, 1, 1)
        plt.plot(t, np.real(self.pulse_shaped_symbols), 'b.-', label = 'Real')
        plt.stem(
            t[symbol_indices],
            np.real(self.pulse_shaped_symbols[symbol_indices]),
            linefmt='b--',
            markerfmt=' ',  # No marker at the head
            basefmt=" ",
            label = "symbol times"
            )

        plt.legend()
        plt.title('Pulse Shaped I (Real Part)')
        plt.subplot(2,1,2)
        plt.plot(t, np.imag(self.pulse_shaped_symbols), 'r.-', label = 'Imaginary')
        plt.stem(
            t[symbol_indices],
            np.imag(self.pulse_shaped_symbols[symbol_indices]),
            linefmt='r--',
            markerfmt=' ',  # No marker at the head
            basefmt=" ",
            label = "symbol times"
        )
        plt.legend()
        plt.title('Pulse Shaped Q (Imaginary Part)')
        plt.tight_layout()
        plt.savefig('media/tx_pulse_shaped_bits.png', dpi=300)
        plt.close()
        # Compute FFT of the baseband pulseshaped signal
        # Compute FFT of the baseband pulse-shaped signal
        pulse_shaped = self.pulse_shaped_symbols
        N = len(pulse_shaped)
        fft_vals = np.fft.fftshift(np.fft.fft(pulse_shaped))
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/self.sample_rate))

        plt.figure(figsize=(10, 6))
        plt.plot(freqs / 1e6, 20 * np.log10(np.abs(fft_vals) ))
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("FFT of Pulse Shaped Baseband Signal")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('media/tx_pulse_shaped_fft.png', dpi=300)
        plt.close()
        
        
        #plot the modulated signal 
        plt.figure(figsize=(10, 6))
        plt.plot(t, np.real(self.qpsk_signal))
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Snippet of Modulated Signal")
        plt.tight_layout()
        plt.savefig('media/tx_waveform_snippet.png', dpi=300)
        plt.close()

        
        #


        #