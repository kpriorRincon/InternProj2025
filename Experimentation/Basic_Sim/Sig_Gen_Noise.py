class SigGen:

    def __init__(self, freq=1.0, amp=1.0, sample_rate=100000, symbol_rate=1000):
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
        # Mean of the noise distribution
        mean_noise = 0
        # Standard deviation of the noise distribution
        std_noise = noise_power

        # Generate noise
        noise_real = np.random.normal(
            mean_noise, std_noise/np.sqrt(2), len(sinusoid))
        noise_imag = np.random.normal(
            mean_noise, std_noise/np.sqrt(2), len(sinusoid))
        noise = noise_real + 1j*noise_imag
        # returns the sinusoid with added noise
        return sinusoid + noise

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
        # Normalize the filter to unit energy
        return t, h 

    def generate_qpsk(self, bits, bool_noise, noise_power=0.01):
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
        # Each symbol is held constant for samples_per_symbol duration

        upsampled_symbols = np.zeros(len(symbols)*samples_per_symbol, dtype = complex)
        upsampled_symbols[::samples_per_symbol] = symbols

        # Root raised cosine filter implementation
        beta = 0.3
        _, pulse_shape = self.rrc_filter( beta, 301, 1/self.symbol_rate, self.sample_rate)
        signal = np.convolve(upsampled_symbols, pulse_shape, 'same')

        # Generate complex phasor at carrier frequency
        phasor = np.exp(1j * 2 * np.pi * self.freq * t)

        # Modulate: multiply upsampled symbols by phasor
        qpsk_waveform = signal * phasor

        # get vertical lines
        t_vertical_lines = []  # Initialize vertical lines for debugging
        for i, symbol in enumerate(symbols):
            # compute the phase offset for the symbol
            phase_offset = np.angle(symbol)
            # debugging print statement to show the symbol and phase offset
            # print(f"Symbol {i}: {symbol}, Phase Offset: {phase_offset}");
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
        message_binary = ''.join(str(bit) for bit in start_sequence) + \
            message_binary + ''.join(str(bit) for bit in end_sequence)
        # print(message_binary)
        # Convert string input to list of integers
        bit_sequence = [int(bit) for bit in message_binary.strip()]
        return bit_sequence
