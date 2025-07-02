import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from binary_search_caf import *
class Detector:
    def __init__(self, N, Ts, beta=0.35, fs=2.4e6, sps=2):
        self.beta = beta
        self.N = N
        self.Ts = Ts
        self.fs = fs
        self.threshold = 8
        self.sps = sps

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
        # samples = (samples - np.min(samples)) / (np.max(np.abs(samples)) - np.min(samples))
        samples = coarse_freq_recovery(samples)
        # default returns 
        start = 0
        end = len(samples) - 1
        detected = False
        
        # find the correlated signal
        # start cor
        cor_start = np.abs(sig.fftconvolve(samples, np.conj(np.flip(match_start)), mode='same'))
        # end cor
        cor_end = np.abs(sig.fftconvolve(samples, np.conj(np.flip(match_end)), mode='same'))
        
        # trim the fat
        trim_factor = 5000

        # get start and end indices
        start = np.argmax(cor_start) - int(len(match_start) / 2)    # go back length of the start/end sequence
        start_idx = np.argmax(cor_start)
        end = np.argmax(cor_end) + int(len(match_end) / 2)
        end_idx = np.argmax(cor_end)
    
        print("Start index: ", start)
        print("End index: ", end)
        
        # select training samples
        samples1 = cor_start[0:start]
        print("Samples1 length:", len(samples1))
        samples2 = cor_start[start + len(match_start):len(samples)-1]
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
        print(f"Max correlation value: {max(np.abs(cor_start))}, Threshold: {self.threshold}")
        if max(np.abs(cor_start)) > self.threshold:
            # if the maximum energy of the correlated signal is greater than the threshold update end index
            if max(cor_end) > self.threshold:
                detected = True
                print("Length of start sequence: ", len(match_start))
                print("Length of end sequence: ", len(match_end))
                #plt.plot(np.fft.fftfreq(len(cor_start), 1/self.fs), 20*np.log10(np.fft.fft(cor_start)), label='Start Correlation')
                #plt.plot(np.fft.fftfreq(len(cor_end), 1/self.fs), 20*np.log10(np.fft.fft(cor_end)), label='End Correlation')
                plt.subplot(2, 1, 1)
                plt.title('Correlation of the Matched Filters')
                plt.plot(np.abs(cor_start), label='start')
                plt.grid()
                plt.legend()
                plt.axhline(y = self.threshold, linestyle = '--', color = 'g')
                #plt.axvline(y = start, linestyle = '--', color = 'r')
                plt.scatter(start_idx, np.abs(cor_start[start_idx]), s = 100, c = 'r', marker = '.')
                
                plt.subplot(2, 1, 2)
                plt.plot(np.abs(cor_end), label='end')
                plt.grid()
                plt.legend()
                plt.axhline(y = self.threshold, linestyle = '--', color = 'g')
                plt.scatter(end_idx, np.abs(cor_end[end_idx]), s = 100, c = 'r', marker = '.')
                #plt.axvline(x = end, linestyle = '--', color = 'r')
                plt.show()

        # if the start index is greater than the end index signal not found, return default values
        if start_idx > end_idx or start < 0 or end > len(samples) or (end - start) > 2 * len(match_start):
            print("Start index greater than end...\nSignal not found...\nSet to defaults")
            start = 0
            end = len(samples) - 1
            detected = False
        
        return detected, start + trim_factor, end + trim_factor
