from Sig_Gen import SigGen, rrc_filter
import numpy as np



fs = 2.88e6
symb_rate = fs/20
freq_offset = 20e3
max_freq = 200
min_freq = -200
start_sequence = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1,
 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1,
 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]

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


def coarse_freq_recovery(qpsk_wave, order=4):

    qpsk_wave_r = qpsk_wave**4

    fft_vals = np.fft.fftshift(np.abs(np.fft.fft(qpsk_wave_r)))
    freqs = np.linspace(-fs/2, fs/2, len(fft_vals))

    freq_tone = freqs[np.argmax(fft_vals)] / order 
    
    t = np.arange(len(qpsk_wave)) / fs
    fixed_qpsk = qpsk_wave * np.exp(-1j*2*np.pi*freq_tone*t)
    
    return fixed_qpsk, freq_tone



def mixing(signal, f):
    t = np.arange(len(signal)) / fs
    return signal * np.exp(1j * 2 * np.pi * f * t)

def correlate(signal, match_filter):
    # Convolve filter with signal and extract highest correlation
    energy_map = np.convolve(signal, np.conj(np.flip(match_filter)), mode = 'same')
    highest_correlation = np.max(np.abs(energy_map))    

    return highest_correlation

visited_freqs = []
correlation_values = []
def binary_search(rx_signal, match_filter, l_freq, r_freq):
    #time.sleep(0.5)
    # Base case: 
    if l_freq >= r_freq:
        return l_freq
    
    # get middle index of each segment when test frequency divided into 2
    l_match_filter = mixing(match_filter, l_freq)
    r_match_filter = mixing(match_filter, r_freq)

    l_energy = correlate(rx_signal, l_match_filter)
    r_energy = correlate(rx_signal, r_match_filter)

    visited_freqs.extend([l_freq, r_freq])
    correlation_values.extend([l_energy, r_energy])

    #print(f"Energy at {l_freq} Hz: {l_energy}. Energy at {r_freq} Hz: {r_energy}.")
    if r_energy > l_energy:
        l_freq += (r_freq - l_freq) // 2 + 1
        freq = binary_search(rx_signal, match_filter, l_freq, r_freq)
    
    elif r_energy < l_energy:
        #print("Should be here")
        r_freq -= (r_freq - l_freq) // 2 + 1
        freq = binary_search(rx_signal, match_filter, l_freq, r_freq)

    else:
        #print("Here")
        # Offset exactly in between, choose middle freq
        freq = l_freq + (r_freq - l_freq) // 2
    
    return freq


def cross_corr_caf(rx_signal):
    # Generate QPSK wave of start marker
    sig_gen = SigGen(0, 1.0, fs, symb_rate)    
    _, marker_wave = sig_gen.generate_qpsk(start_sequence)

    # Impulse response of RRC filter
    _, h = rrc_filter(0.4, 301, 1/symb_rate, fs)
    delay = (len(h) - 1) // 2

    # RRC start marker match filter
    #rrc_start_filter = np.convolve(marker_wave, h, mode = 'full')    
    #rrc_start_filter = rrc_start_filter[delay: delay + len(marker_wave)]
    
    #strt = time.time()
    # Binary search for frequency offset
    freq_found = binary_search(rx_signal, marker_wave, min_freq, max_freq)

    #print(f"Total time for binary search: {time.time() - strt} s.")
    return freq_found


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
    sig_gen = SigGen(freq=900e6, amp=1,sample_rate=fs, symbol_rate=symb_rate)
    bits = sig_gen.message_to_bits('hello there' * 3)
    t, qpsk_wave = sig_gen.generate_qpsk(bits)

    # Set frequency Offset
    qpsk_wave = qpsk_wave * np.exp(1j* 2 * np.pi * freq_offset * t) # shifts down by freq offset
    
    #Tune down to baseband
    tuned_sig = qpsk_wave * np.exp(-1j * 2 * np.pi * sig_gen.freq * t)
    coarse_fixed_sig, coarse_freq = coarse_freq_recovery(tuned_sig)

    print(f"Coarse Frequency Correction: {coarse_freq} Hz")

    caf_freq = cross_corr_caf(coarse_fixed_sig)
    caf_fixed_sig = coarse_fixed_sig * np.exp(-1j * 2 * np.pi * caf_freq * t)

    print(f"Binary Search CAF Frequency Correction: {caf_freq} Hz")

    best_alpha = None
    best_beta = None
    min_error = float('inf')
    best_correction = None

    alphas = np.arange(0, 0.2, 0.001)
    betas = np.arange(0, 0.01, 0.00001)

    print(f"Testing {len(alphas)} alphas and {len(betas)} betas.")
    print(f"Total number of test cases: {len(alphas) * len(betas)}.")
    #print(costas_loop(coarse_fixed_sig, 0.159, 0.00881))
    
    counter = 0
    for a in alphas:
        for b in betas:
            fine_freq = costas_loop(caf_fixed_sig, alpha=a, beta=b)
            total_correction = coarse_freq + caf_freq + fine_freq
            error = abs(freq_offset - total_correction)
            #compare to freq_offset
            print(f"Test Case: {counter}", end='\r')
            counter += 1
            if error < min_error:
                min_error = error
                best_alpha = a
                best_beta = b
                best_correction = fine_freq

    print(f"Frequency Offset: {freq_offset} Hz")    
    print(f"Best alpha: {best_alpha}")
    print(f"Best beta: {best_beta}")
    print(f"Best Costas Correction {best_correction} Hz")
    print(f"Total correction: {best_correction + caf_freq + coarse_freq} Hz")
    print(f"Minimum error: {min_error} Hz")

if __name__ == "__main__":
    main()
