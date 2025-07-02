import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

class Detector:
    def __init__(self, marker_start, marker_end, N, Ts, beta=0.35, fs=2.4e6, sps=2):
        self.marker_start = marker_start
        self.marker_end = marker_end
        self.beta = beta
        self.N = N
        self.Ts = Ts
        self.fs = fs
        self.threshold = 8
        self.sps = sps

    def step_detect(self, samples):
        """
        Detect the signal with a step function

        Parameters:
        - samples: samples read in by the RTL-SDR

        Returns:
        - detected: bool set high when signal found
        - start: start index
        - end: end index

        """
        t = len(samples) - 1
        
        # make step functoin
        step_function = [0]*20000 + [1]*20000
        step_function = np.array(step_function)
        
        # correlation signal with step function
        correlation = sig.fftconvolve(samples, step_function)
        
        # plot
        plt.plot(correlation)
        #plt.show()

    def detector(self, samples, match_start, match_end):
        """
        Detects the sent signal amongst noise

        Parameters:
        - samples: samples read in by the RTL-SDR
        - match_start: match filter for the start sequence
        - match_end: match filter for the end sequence

        Returns:
        - detected: bool saying a match was found
        - start: start index of the message
        - end: end index of the message
        """
        print("Length of samples: ", len(samples))
        # normalize the samples
        samples = (samples - np.min(samples)) / (np.max(np.abs(samples)) - np.min(samples))

        # default returns 
        start = 0
        end = len(samples) - 1
        detected = False
        
        # find the correlated signal
        # start cor
        cor_start = sig.fftconvolve(samples**4, np.conj(np.flip(match_start)**4), mode='same')
        # start cor normalize
        cor_start = (cor_start - np.min(cor_start)) / (np.max(np.abs(cor_start)) - np.min(cor_start))
        # end cor
        cor_end = sig.fftconvolve(samples**4, np.conj(np.flip(match_end)**4), mode='same')
        # end cor normalize
        cor_end = (cor_end - np.min(cor_end)) / (np.max(np.abs(cor_end)) - np.min(cor_end))
        
        # trim the fat
        trim_factor = 5000
        start_trimmed = cor_start[trim_factor:len(cor_start)-trim_factor]
        end_trimmed = cor_end[trim_factor:len(cor_end)-trim_factor]

        # get start and end indices
        start = np.argmax(np.abs(start_trimmed)) - int(len(match_start) / 2)    # go back length of the start/end sequence
        start_idx = np.argmax(np.abs(start_trimmed))
        end = np.argmax(np.abs(end_trimmed)) + int(len(match_end) / 2)
        end_idx = np.argmax(np.abs(end_trimmed))

        print("Start index: ", start)
        print("End index: ", end)
        
        # select training samples
        samples1 = start_trimmed[0:start]
        print("Samples1 length:", len(samples1))
        samples2 = start_trimmed[start + len(match_start):len(samples)-1]
        print("Samples2 length:", len(samples2))
        training_samples = np.concatenate([samples1, samples2])

        # calculate the threshold
        M = len(training_samples)
        print(f"Training samples length: {M}")
        if M > 0:
            P_fa = 0.075 # probability of false alarm
            alpha = (P_fa**(-1/M) - 1) * M
            Pn = np.sum(np.abs(training_samples)) / M
            self.threshold = Pn * alpha

        # if the maximum energy of the correlated signal is greater than the threshold update start index
        print(f"Max correlation value: {max(np.abs(start_trimmed))}, Threshold: {self.threshold}")
        if max(np.abs(start_trimmed)) > self.threshold:
            # if the maximum energy of the correlated signal is greater than the threshold update end index
            if max(np.abs(end_trimmed)) > self.threshold:
                detected = True
                print("Length of start sequence: ", len(match_start))
                print("Length of end sequence: ", len(match_end))
                #plt.plot(np.fft.fftfreq(len(cor_start), 1/self.fs), 20*np.log10(np.fft.fft(cor_start)), label='Start Correlation')
                #plt.plot(np.fft.fftfreq(len(cor_end), 1/self.fs), 20*np.log10(np.fft.fft(cor_end)), label='End Correlation')
                plt.subplot(2, 1, 1)
                plt.title('Correlation of the Matched Filters')
                plt.plot(np.abs(start_trimmed), label='start')
                plt.grid()
                plt.legend()
                plt.axhline(y = self.threshold, linestyle = '--', color = 'g')
                #plt.axvline(y = start, linestyle = '--', color = 'r')
                plt.scatter(start_idx, np.abs(start_trimmed[start_idx]), s = 100, c = 'r', marker = '.')
                
                plt.subplot(2, 1, 2)
                plt.plot(np.abs(end_trimmed), label='end')
                plt.grid()
                plt.legend()
                plt.axhline(y = self.threshold, linestyle = '--', color = 'g')
                plt.scatter(end_idx, np.abs(end_trimmed[end_idx]), s = 100, c = 'r', marker = '.')
                #plt.axvline(x = end, linestyle = '--', color = 'r')
                plt.show()

        # if the start index is greater than the end index signal not found, return default values
        if start_idx > end_idx or start < 0 or end > len(samples) or (end - start) > 2 * len(match_start):
            print("Start index greater than end...\nSignal not found...\nSet to defaults")
            start = 0
            end = len(samples) - 1
            detected = False
        
        return detected, start + trim_factor, end + trim_factor
