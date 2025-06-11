import numpy as np 
import Sig_Gen_Noise as SigGen
from scipy import constants
import matplotlib.pyplot as plt
def applyAWGN(signal, snr):
    """
    Adds Additive White Gaussian Noise (AWGN) to a complex input signal based on the specified Signal-to-Noise Ratio (SNR).

    Parameters:
        signal (np.ndarray): Input signal array (can be real or complex).
        snr (float): Desired signal-to-noise ratio in dB.

    Returns:
        np.ndarray: Noisy signal with AWGN added.

    Notes:
        - The function assumes the input signal is a NumPy array.
        - The SNR is specified in decibels (dB).
        - The noise is generated as complex Gaussian noise.
    """

    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr/10)
    noise_power = signal_power/snr_linear
    sigma = np.sqrt(noise_power)/2
    noise = np.random.normal(0, sigma,  len(signal)) + 1j*np.random.normal(0, sigma,  len(signal))
    return signal + noise

def applyFreqShift(signal, relative_velocity_satelite):

    pass

def applyTimeDelay(signal, sat_dist, sample_rate):
    """
    Applies a time delay to the input signal based on satellite distance and sample rate.

    Parameters:
        signal (np.ndarray): Input signal array (can be real or complex).
        sat_dist (float): Distance to the satellite in meters.
        sample_rate (float): Sampling rate in Hz.

    Returns:
        np.ndarray: Time-delayed signal.
    """
    #get the time delay
    c = constants.speed_of_light
    time_delay = sat_dist/c #m/m/s = s
    #we need to shift the signal forward in samples
    delay_samples = int(np.round(time_delay * sample_rate))
    #apply the delay
    singal_delayed = np.concatenate([np.zeros(delay_samples, dtype=complex), signal])
    return singal_delayed

def applyPhaseShift(signal):
    pass

def applyIQImbalence(signal):
    pass

def Channel_Model(signal, snr, satellite_velocity, satellite_distance, ):
    #the channel model will take in a signal and apply channel effects
    pass

#test


bit_sequence = np.random.randint(0, 2, 48)
sig_gen =SigGen.SigGen(freq=900e6, amp=1.0, sample_rate =20000, symbol_rate = 1000)
t, qpsk_waveform, symbols, t_vertical_lines, upsampled_symbols = sig_gen.generate_qpsk(bit_sequence, False, 0)
plt.plot(t, upsampled_symbols)
plt.show()
#print(bit_sequence)

# plt.figure()
# plt.plot(t, sine)

# plt.figure()
# noisy_sig = applyAWGN(sine, 1)
# plt.plot(t, noisy_sig)