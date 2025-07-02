from Sig_Gen import SigGen, rrc_filter
import numpy as np
from config import *
from scipy.signal import fftconvolve, resample_poly
import time
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fs = 2.88e6
symb_rate = fs/20
freq_offset = 20e3
max_freq = 200
min_freq = -200
snr_db = 20

def phase_detector_4(sample):
    if sample.real > 0:
        a = 1.0
    else:
        a = -1.0
    if sample.imag > 0:
        b = 1.0
    else:
        b = -1.0
    return a * sample.imag - b * sample.real

def mixing(signal, f, ip_val=1):
    t = np.arange(len(signal)) / (SAMPLE_RATE * ip_val)
    return signal * np.exp(1j * 2 * np.pi * f * t)

def correlate(signal, match_filter):
    # Convolve filter with signal and extract highest correlation
    energy_map = fftconvolve(signal, np.conj(np.flip(match_filter)), mode = 'same')
    highest_correlation = np.max(np.abs(energy_map))    

    return highest_correlation

visited_freqs = []
correlation_values = []
def binary_search(rx_signal, match_filter, l_freq, r_freq):
    # Base case: 
    if l_freq >= r_freq:
        return l_freq
    
    # get middle index of each segment when test frequency divided into 2
    l_match_filter = mixing(match_filter, l_freq, INTERPOLATION_VAL)
    r_match_filter = mixing(match_filter, r_freq, INTERPOLATION_VAL)

    l_energy = correlate(rx_signal, l_match_filter)
    r_energy = correlate(rx_signal, r_match_filter)

    visited_freqs.extend([l_freq, r_freq])
    correlation_values.extend([l_energy, r_energy])

    print(f"Energy at {l_freq} Hz: {l_energy}. Energy at {r_freq} Hz: {r_energy}.")
    if r_energy > l_energy:
        l_freq += (r_freq - l_freq) // 2 + 1
        freq = binary_search(rx_signal, match_filter, l_freq, r_freq)
    
    elif r_energy < l_energy:
        r_freq -= (r_freq - l_freq) // 2 + 1
        freq = binary_search(rx_signal, match_filter, l_freq, r_freq)

    else:
        # Offset exactly in between, choose middle freq
        freq = l_freq + (r_freq - l_freq) // 2
    
    return freq


def cross_corr_caf(rx_signal):
    # Generate QPSK wave of start marker
    sig_gen = SigGen(0, 1.0)    
    _, marker_filter = sig_gen.generate_qpsk(START_MARKER)

    #Interpolate signal and match filter
    ip_signal = resample_poly(rx_signal, INTERPOLATION_VAL, 1)
    ip_filter = resample_poly(marker_filter, INTERPOLATION_VAL, 1)

    strt = time.time()
    # Binary search for frequency offset
    freq_found = binary_search(ip_signal, ip_filter, min_freq, max_freq)

    print(f"Total time for binary search: {time.time() - strt} s.")
    print(f"Binary Search CAF: {freq_found}")

    #Correlate one last time to get index
    up_mixed_filter = mixing(ip_filter, freq_found, INTERPOLATION_VAL)
    start_map = fftconvolve(ip_signal, np.conj(np.flip(up_mixed_filter)), mode = 'same')
    start_idx = np.argmax(np.abs(start_map)) - int((32) * (SAMPLE_RATE * INTERPOLATION_VAL / SYMB_RATE))


    #Correlate with end marker match filter for end idx
    _, end_filter = sig_gen.generate_qpsk(END_MARKER)
    ip_end_filter = resample_poly(end_filter, INTERPOLATION_VAL, 1)
    mixed_end_filter = mixing(ip_end_filter, freq_found, INTERPOLATION_VAL)
    end_map = fftconvolve(ip_signal, np.conj(np.flip(mixed_end_filter)), mode = 'same')
    end_idx = np.argmax(np.abs(end_map)) + int((32) * (SAMPLE_RATE * INTERPOLATION_VAL / SYMB_RATE))


    # Reslice signal
    print(f"Start: {start_idx} End: {end_idx}")
    deci_signal = ip_signal[start_idx: end_idx:16]   


    t = np.arange(len(deci_signal)) / SAMPLE_RATE
    fixed_signal = deci_signal * np.exp(-1j * 2 * np.pi * freq_found * t)

    return fixed_signal, freq_found


def coarse_freq_recovery(qpsk_wave, order=4):

    qpsk_wave_r = qpsk_wave**4

    fft_vals = np.fft.fftshift(np.abs(np.fft.fft(qpsk_wave_r)))
    freqs = np.linspace(-fs/2, fs/2, len(fft_vals))

    freq_tone = freqs[np.argmax(fft_vals)] / order 
    
    t = np.arange(len(qpsk_wave)) / fs
    fixed_qpsk = qpsk_wave * np.exp(-1j*2*np.pi*freq_tone*t)
    
    return fixed_qpsk, freq_tone


def costas_loop(qpsk_wave, alpha, beta):
    # requires downconversion to baseband first
    N = len(qpsk_wave)
    phase = 0
    freq = 0 # derivative of phase; rate of change of phase (radians/sample)
    #Following params determine feedback loop speed
    #alpha = 0.002#0.0006 #0.132 immediate phase correction based on current error
    #beta = 0.000000634#0.0000004 #0.00932  tracks accumalated phase error
    out = np.zeros(N, dtype=np.complex64)
    freq_log = []
    
    for i in range(N):
        out[i] = qpsk_wave[i] * np.exp(-1j*phase) #adjust input sample by inv of estimated phase offset
        error = phase_detector_4(out[i])

        freq += (beta * error)
        #log frequency in Hz
        freq_log.append(freq * fs / (2 * np.pi))
        phase += freq + (alpha * error)

        while phase >= 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi
    
    return freq_log[-1]



def main():

    #Generate QPSK at carrier frequency
    sig_gen = SigGen(freq = 900e6, amp = 1)
    bits = sig_gen.message_to_bits('hello there ' * 3)
    t, qpsk_wave = sig_gen.generate_qpsk(bits)

    # Set frequency Offset
    qpsk_wave = qpsk_wave * np.exp(1j* 2 * np.pi * freq_offset * t) # shifts up by freq offset
    
    #Tune down to baseband
    tuned_sig = qpsk_wave * np.exp(-1j * 2 * np.pi * sig_gen.freq * t)

    coarse_fixed_sig, coarse_freq = coarse_freq_recovery(tuned_sig)

    print(f"Coarse Frequency Correction: {coarse_freq} Hz")

    sig, frequency_CAF = cross_corr_caf(coarse_fixed_sig)
    print(f'CAF frequency Correction: {frequency_CAF} Hz')
    
    best_alpha = None
    best_beta = None
    min_error = float('inf')
    best_correction = None

    alphas = np.arange(0, 0.01, 0.001) #step 0.00005
    betas = np.arange(0, 0.00001, 0.000001)#0.00000005

    print(f"Testing {len(alphas)} alphas and {len(betas)} betas.")
    print(f"Total number of test cases: {len(alphas) * len(betas)}.")

    error_matrix = []
    counter = 0
    for a in alphas:
        error_row = []
        for b in betas:
            fine_freq = costas_loop(sig, alpha=a, beta=b)
            total_correction = coarse_freq + frequency_CAF + fine_freq 
            error = abs(freq_offset - total_correction)
            #compare to freq_offset
            print(f"Test Case: {counter}", end='\r')
            counter += 1
            if error < min_error:
                min_error = error
                best_alpha = a
                best_beta = b
                best_correction = fine_freq
            
            error_row.append(error)
        error_matrix.append(error_row)

    error_matrix = np.array(error_matrix)

    print(f"Best alpha: {best_alpha}")
    print(f"Best beta: {best_beta}")
    print(f"Best Costas Correction {best_correction} Hz")
    print(f"Total correction: {best_correction + coarse_freq} Hz")
    print(f"Minimum error: {min_error} Hz")
    
    A, B = np.meshgrid(betas, alphas)  # Note: betas = X, alphas = Y
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, error_matrix, cmap='viridis')
    ax.set_xlabel('Beta')
    ax.set_ylabel('Alpha')
    ax.set_zlabel('Error')
    ax.set_title('Costas Loop Error Surface')
    plt.show()
    plt.figure(figsize=(10, 6))
    sns.heatmap(error_matrix, xticklabels=np.round(betas, 8), yticklabels=np.round(alphas, 5), cmap='mako', annot=False)
    plt.xlabel('Beta')
    plt.ylabel('Alpha')
    plt.title('Costas Loop Error Heatmap')
    plt.show()


if __name__ == "__main__":
    main()
