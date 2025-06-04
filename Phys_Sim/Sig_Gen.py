class SigGen:
    
    def __init__(self, sample_rate, symbol_rate):
        import numpy as np
        self.freq = None  # Frequency in Hz
        self.sample_rate = sample_rate  # sample rate in samples per second
        self.symbol_rate = symbol_rate  # Symbol rate in symbols per second about 30% of the frequency
        self.amp = None    # Amplitude

        self.time_vector = None
        self.qpsk_waveform = None
        self.time_vertical_lines = None
        self.symbols = None
        # Map bit pairs to complex symbols
        self.mapping = {
            (0, 0): (1 + 1j) / np.sqrt(2),
            (0, 1): (-1 + 1j) / np.sqrt(2),
            (1, 1): (-1 - 1j) / np.sqrt(2),
            (1, 0): (1 - 1j) / np.sqrt(2)
        }

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
        symbols = []
        if len(bits) % 2 != 0:
            raise ValueError("Bit sequence must have an even length.")

        #seperate into odd and even and map to complex symbols
        symbols = [self.mapping[(bits[i], bits[i + 1])] for i in range(0, len(bits), 2)]

        t_vertical_lines = []  # Initialize vertical lines for debugging
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        #time vector for the wave form defined start at 0 end at the length of the symbols times samples per symbol
        # and spaced by the sample rate
        t = np.arange(0, len(symbols) * samples_per_symbol) / self.sample_rate

        # Initialize the QPSK waveform
        qpsk_waveform = np.zeros_like(t)
        #for loop that tracks the index of the symbol and the symbol itself
        for i, symbol in enumerate(symbols):
            #compute the phase offset for the symbol
            phase_offset = np.angle(symbol)
            #debugging print statement to show the symbol and phase offset
            #print(f"Symbol {i}: {symbol}, Phase Offset: {phase_offset}");
            idx_start = i * samples_per_symbol
            idx_end = idx_start + samples_per_symbol
            time_slice = t[idx_start:idx_end] # time slice for the current symbol
            qpsk_waveform[idx_start:idx_end] = (
                #eqn from Haupt pg 86
                1/np.sqrt(2)*np.cos(2*self.freq*np.pi * time_slice + phase_offset)
                #add vertical dashed lines at time slices of the symbols
            )
            t_vertical_lines.append(idx_start/self.sample_rate)

        self.time_vector = t
        self.qpsk_waveform = qpsk_waveform
        self.time_vertical_lines = t_vertical_lines
        self.symbols = symbols
    
    def message_to_bits(self, message):
        """
        Author: Skylar Harris
        Convert a string message to a list of bits.
        
        Parameters:
            message (str): The input string message.
        
        Returns:
            list: List of bits (0s and 1s).
        """
        # prefix the message with the letter 'R' as a marker
        message = 'R' + message
        message_binary = ''.join(format(ord(x), '08b') for x in message)
        # print(message_binary)
        # Convert string input to list of integers
        bit_sequence = [int(bit) for bit in message_binary.strip()]
        return bit_sequence

    def plot_time_png(self,message):
        "this function plots 3 sections of the qpsk wave form in the time domain"
        " and stores them into png files in the qpsk_sig_gen folder'"
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(15, 5))
        plt.plot(self.time_vector, self.qpsk_waveform)
        plt.ylim(-1/np.sqrt(2)*self.amp-.5, 1/np.sqrt(2)*self.amp+.5)

        #if there are more than 10 symbols only show the first ten symbols
        if len(self.time_vertical_lines) > 10:
            plt.xlim(0, 10/self.symbol_rate)  # Show first 10 symbol periods
        #if not don't touch the xlim

        for lines in self.time_vertical_lines:
            #add vertical lines at the symbol boundaries
            if len(self.time_vertical_lines) > 10:
                if lines < 9/self.symbol_rate:
                    plt.axvline(x=lines, color='black', linestyle='--', linewidth=1)

                    #add annotation for the symbol e.g. '00', '01', '10', '11'
                    # Reverse mapping: symbol -> binary pair
                    symbol = self.symbols[self.time_vertical_lines.index(lines)]
                    # Reverse the mapping to get binary pair from symbol
                    reverse_mapping = {v: k for k, v in self.mapping.items()}
                    binary_pair = reverse_mapping.get(symbol, '')
                    formatted_pair =str(binary_pair).replace("(", "").replace(")", "").replace(", ", "")
                    #debug
                    #print(formatted_pair)
                    x_dist = 1 / (2.7 * self.symbol_rate) #half the symbol period 
                    y_dist = 0.707*self.amp + .2 # 0.807 is the amplitude of the QPSK waveform
                    plt.annotate(formatted_pair, xy=(lines, 0), xytext=(lines + x_dist, y_dist), fontsize=17)  
            else:
                if lines < len(self.time_vector):
                    plt.axvline(x=lines, color='black', linestyle='--', linewidth=1)

                    #add annotation for the symbol e.g. '00', '01', '10', '11'
                    # Reverse mapping: symbol -> binary pair
                    symbol = self.symbols[self.time_vertical_lines.index(lines)]
                    # Reverse the mapping to get binary pair from symbol
                    reverse_mapping = {v: k for k, v in self.mapping.items()}
                    binary_pair = reverse_mapping.get(symbol, '')
                    formatted_pair =str(binary_pair).replace("(", "").replace(")", "").replace(", ", "")
                    #debug
                    #print(formatted_pair)
                    x_dist = 1 / (2.7 * self.symbol_rate) #half the symbol period 
                    y_dist = 0.707*self.amp + .2 # 0.807 is the amplitude of the QPSK waveform
                    plt.annotate(formatted_pair, xy=(lines, 0), xytext=(lines + x_dist, y_dist), fontsize=17)
                
        if len(self.time_vertical_lines) > 10:
            plt.title(f'QPSK Waveform for \"{message}\" (first 10 symbol periods)')
        else:
            plt.title(f'QPSK Waveform for \"{message}\"')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()
        # Save the plot to a file
        plt.savefig(f'qpsk_sig_gen/1_qpsk_waveform.png', dpi=300)    
        #print("Debug: plot generated")
        return
    
    def plot_freq_png(self):
        #same figure size as above
        import matplotlib.pyplot as plt
        import numpy as np
        n = len(self.time_vector)
        freqs = np.fft.fftfreq(n, d=1/self.sample_rate)
        # FFT of original and shifted signals
        fft = np.fft.fft(self.qpsk_waveform)
        fft_db = 20 * np.log10(np.abs(fft))
        # get fft of qpsk signal
        
        plt.figure(figsize=(15, 5))
        plt.plot(freqs,fft_db)
        plt.title("FFT of QPSK signal")
        plt.xlim(0, 1000e6)
        plt.ylim(0, np.max(fft_db)+10)
        #save plot
        plt.savefig(f'qpsk_sig_gen/2_qpsk_waveform.png', dpi = 300)

    def handler(self,message, frequency):
        self.freq = frequency
        self.symbol_rate = .01*frequency
        self.generate_qpsk(self.message_to_bits(message))
        self.plot_time_png(message)
        self.plot_freq_png()