import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import resample_poly
from Sig_Gen import SigGen, rrc_filter
DEBUG = 1
freq_offset = 2000
fs = 2.88e6
symb_rate = 2.88e6/20
max_freq = 200
min_freq = -200
interpolation_val = 16

def integer_delay(signal, num_samples):
    '''Apply an integer delay by padding the front of the signal with zeros'''
    signal = np.concatenate([np.zeros(num_samples, dtype=complex), signal])
    t = np.arange(len(signal))/fs 
    return signal, t 

def fractional_delay(t, signal, delay):

    total_delay = delay * fs # seconds * samples/second = samples
    fractional_delay = total_delay % 1 # to get the remainder
    integer_delay = int(total_delay)
    print(f"Integer delay in samples: {integer_delay}")
    print(f'fractional delay in samples: {fractional_delay}')
    #then shift by fractional samples with the leftover 
    delay_in_samples = fractional_delay
    #Fs * delay_in_sec # samples/seconds * seconds = samples
    
    #pad with zeros for the integer_delay
    if integer_delay > 0:
        signal = np.concatenate([np.zeros(integer_delay, dtype=complex), signal])
    
    #filter taps
    N = 301
    #construct filter
    n = np.linspace(-N//2, N//2,N)
    h = np.sinc(n-delay_in_samples)
    h *= np.hamming(N) #something like a rectangular window
    h /= np.sum(h) #normalize to get unity gain, we don't want to change the amplitude/power


    #apply filter: same time-aligned output with the same size as the input
    new_signal = np.convolve(signal, h, mode='full')
    delay = (N - 1) // 2
    new_signal = new_signal[delay:delay+len(signal)]
    new_t = np.arange(len(new_signal)) / fs

    return new_t, new_signal

def phase_offset(signal):
    theta = np.random.uniform(-np.pi, np.pi)
    new_signal = signal * np.exp(1j * theta)
    print(f'Random Phase Offset: {np.rad2deg(theta)}')

    return new_signal

def coarse_freq_recovery(qpsk_wave, order=4):


    qpsk_wave_r = qpsk_wave**4

    fft_vals = np.fft.fftshift(np.abs(np.fft.fft(qpsk_wave_r)))
    freqs = np.linspace(-fs/2, fs/2, len(fft_vals))

    freq_tone = freqs[np.argmax(fft_vals)] / order 
    print(f'frequency offset(coarse freq): {freq_tone}')
    
    t = np.arange(len(qpsk_wave)) / fs
    fixed_qpsk = qpsk_wave * np.exp(-1j*2*np.pi*freq_tone*t)

    return fixed_qpsk



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

def costas_loop(qpsk_wave):
    # requires downconversion to baseband first
    N = len(qpsk_wave)
    phase = 0
    freq = 0 # derivative of phase; rate of change of phase (radians/sample)
    #Following params determine feedback loop speed
    alpha = 0.001#0.027 #0.132 immediate phase correction based on current error
    beta = 5e-7#0.00286 #0.00932  tracks accumalated phase error
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
    
    #finds the frequency at the end when it converged
    print(f'Costas Converged Frequency Offset: {freq_log[-1]}')

    plt.plot(freq_log,'.-')
    plt.title('freq converge')
    plt.show()
    t = np.arange(len(qpsk_wave)) / fs
    return qpsk_wave * np.exp(-1j * 2 * np.pi * freq_log[-1] * t)
    

def mixing(signal, f, ip_val=1):
    t = np.arange(len(signal)) / (fs * ip_val)
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
    l_match_filter = mixing(match_filter, l_freq, interpolation_val)
    r_match_filter = mixing(match_filter, r_freq, interpolation_val)

    l_energy = correlate(rx_signal, l_match_filter)
    r_energy = correlate(rx_signal, r_match_filter)

    visited_freqs.extend([l_freq, r_freq])
    correlation_values.extend([l_energy, r_energy])

    print(f"Energy at {l_freq} Hz: {l_energy}. Energy at {r_freq} Hz: {r_energy}.")
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
    _, marker_filter = sig_gen.generate_qpsk(sig_gen.start_sequence)

    #Interpolate signal and match filter
    ip_signal = resample_poly(rx_signal, interpolation_val, 1)
    ip_filter = resample_poly(marker_filter, interpolation_val, 1)

    strt = time.time()
    # Binary search for frequency offset
    freq_found = binary_search(ip_signal, ip_filter, min_freq, max_freq)

    print(f"Total time for binary search: {time.time() - strt} s.")
    print(f"Binary Search CAF: {freq_found}")

    #Correlate one last time to get index
    up_mixed_filter = mixing(ip_filter, freq_found, interpolation_val)
    start_map = np.convolve(ip_signal, np.conj(np.flip(up_mixed_filter)), mode = 'same')
    start_idx = np.argmax(np.abs(start_map)) - int((32) * (fs * interpolation_val / symb_rate))

    if DEBUG:
        plt.figure()
        plt.title('start correlation')
        plt.plot(np.abs(start_map))
        plt.show()
    #Correlate with end marker match filter for end idx
    _, end_filter = sig_gen.generate_qpsk(sig_gen.end_sequence)
    ip_end_filter = resample_poly(end_filter, interpolation_val, 1)
    mixed_end_filter = mixing(ip_end_filter, freq_found, interpolation_val)
    end_map = np.convolve(ip_signal, np.conj(np.flip(mixed_end_filter)), mode = 'same')
    end_idx = np.argmax(np.abs(end_map)) + int((32) * (fs * interpolation_val / symb_rate))

    if DEBUG:
        plt.figure()
        plt.title('ends correlation')
        plt.plot(np.abs(end_map))
        plt.show()
    
    # Reslice signal
    print(f"Start: {start_idx} End: {end_idx}")
    deci_signal = ip_signal[start_idx: end_idx:16]

    #t = np.arange(len(rx_signal)) / fs
    #t, deci_signal = fractional_delay(t, rx_signal, -start_idx / 16)

    #deci_signal = deci_signal[:int(end_idx / 16)]    

    # Fix phase offset
    sig_start = deci_signal[0: int(64 * fs / symb_rate)]
    h =  sig_start / marker_filter
    h_norm = np.mean(h / np.abs(h))
    print(f'Phase offset found: {np.rad2deg(np.angle(h_norm))}')

    deci_signal /= h_norm

    t = np.arange(len(deci_signal)) / fs
    fixed_signal = deci_signal * np.exp(-1j * 2 * np.pi * freq_found * t)

    return fixed_signal

# Notes: No need for coarse or fine as CAF will correct freq in this test

def main():
    #Generate QPSK at Carrier Frequency
    sig_gen = SigGen(freq=900e6, amp=1,sample_rate=fs, symbol_rate=symb_rate)
    bits = sig_gen.message_to_bits('hello there' * 3)
    t, qpsk_wave = sig_gen.generate_qpsk(bits)    
    print(f"Length of TX Signal: {len(qpsk_wave)}")
    # Integer time delay
    #qpsk_wave, t = integer_delay(qpsk_wave, 100)
    t, qpsk_wave = fractional_delay(t, qpsk_wave, 0.00232)

    # Set frequency offset
    qpsk_wave = qpsk_wave * np.exp(1j* 2 * np.pi * freq_offset * t) # shifts down by freq offset
    
    # Uniformly random phase offset
    qpsk_wave = phase_offset(qpsk_wave)

    #Tune down to baseband
    qpsk_base = qpsk_wave * np.exp(-1j * 2 * np.pi * sig_gen.freq * t)

    coarse_fixed_sig = coarse_freq_recovery(qpsk_base)

    # Run CAF and return frequency offset found with highest correlation
    caf_fixed_sig = cross_corr_caf(coarse_fixed_sig)

    plt.plot(range(len(visited_freqs)), visited_freqs, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Frequency Offset (Hz or bins)")
    plt.title("Binary Search Convergence on Frequency Offset")
    plt.grid(True)
    plt.show()

    #Down convert with offset
    final_fixed_sig = costas_loop(caf_fixed_sig)


    #freq_fixed_signal = cross_corr(fine_fixed_sig)


    # Pass through RRC filter
    _, h = rrc_filter(0.4, 301, 1/symb_rate, fs)
    delay = (301 - 1) // 2 
    signal_ready = np.convolve(caf_fixed_sig, h, mode = 'full')    
    signal_ready = signal_ready[delay: delay + len(final_fixed_sig)]

    signal_ready = signal_ready[::int(fs/symb_rate)]

    #convert bits to string
    bits = np.zeros((len(signal_ready), 2), dtype=int)
    for i in range(len(signal_ready)):
        angle = np.angle(signal_ready[i], deg=True) % 360

        # codex for the phases to bits
        if 0 <= angle < 90:
            bits[i] = [0, 0]  # 45°
        elif 90 <= angle < 180:
            bits[i] = [0, 1]  # 135°
        elif 180 <= angle < 270:
            bits[i] = [1, 1]  # 225°
        else:
            bits[i] = [1, 0]  # 315°
    
    #concatennate these lists of lists
    bits = bits.flatten().tolist()
    
    bits = bits[128:-128]
    bits = ''.join(str(bit) for bit in bits)
    # convert the bits into a string
    
    decoded_string = ''.join(chr(int(bits[i*8:i*8+8],2)) for i in range(len(bits)//8))
    print(f"The decoded message = {decoded_string}")

if __name__ == "__main__":
    main()