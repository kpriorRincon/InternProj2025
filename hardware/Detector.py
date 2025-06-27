import numpy as np
import scipy.signal as sig

class Detector:
    def __init__(self, marker_start, marker_end, beta, N, Ts, fs):
        self.marker_start = marker_start
        self.marker_end = marker_end
        self.beta = beta
        self.N = N
        self.Ts = Ts
        self.fs = fs
        self.threshold = 0

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
        t = np.arange(-N // 2, N // 2 + 1) / fs 

        h = np.zeros_like(t) 

        for i in range(len(t)):
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
        return t, h

    def matchedFilter(self, sps):
        """
        Generate the matched filter for correlation detection

        Parameters:
        - sps: samples per symbol

        Returns:
        - match_start: returns the matched filter for the start sequence
        - metch_end: returns the matched filter for the end sequence
        """
        # upsample
        L = sps - 1
        up_start = np.zeros(0, len(self.marker_start)*sps)
        up_start[::L] = self.marker_start
        up_end = np.zeros(0, len(self.marker_end)*sps)
        up_end[::L] = self.marker_end

        # filter coefficients
        
        _, h = self.rrc_filter(self.beta, self.N, self.Ts, self.fs)

        # pulse shape
        match_start = sig.fftconvolve(h, up_start)
        match_end = sig.fftconvolve(h , up_end)

        return match_start, match_end
    
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
        # default returns 
        start = 0
        end = len(samples) - 1
        detected = False
        
        # updated best guess for the power threshold 
        self.threshold = (self.threshold + np.sum(np.abs(samples)) / len(samples)) / 2

        # find the correlated signal
        cor_start = sig.fftconvolve(samples, np.conj(np.flip(match_start)), mode='same')
        
        # if the maximum energy of the correlated signal is greater than the threshold update start index
        if max(np.abs(cor_start)) > self.threshold:
            cor_end = sig.fftconvolve(samples, np.conj(np.flip(match_end)), mode='same')
            start = np.argmax(cor_start) - self.sps*16
            
            # if the maximum energy of the correlated signal is greater than the threshold update end index
            if max(np.abs(cor_end)) > self.threshold:
                end = np.argmax(cor_end)
                detected = True

        # if the start index is greater than the end index signal not found, return default values
        if start > end:
            print("Start index greater than end...\nSignal not found...\nSet to defaults")
            start = 0
            end = len(samples) - 1
            detected = False
        
        return detected, start, end