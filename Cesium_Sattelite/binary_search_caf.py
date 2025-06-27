import numpy as np
import matplotlib.pyplot as plt
import time
from Sig_Gen import SigGen, rrc_filter
freq_offset = 21
fs = 2.88e6
symb_rate = 2.88e6/20
max_freq = 200
min_freq = -200
start_sequence = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1,
 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1,
 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]


def integer_delay(qpsk_wave, int_delay):
    padded_signal = np.pad(qpsk_wave, (int_delay, 0), mode='constant')
    t = np.arange(len(padded_signal)) / fs
    return padded_signal , t

def phase_offset(signal):
    theta = np.random.uniform(-np.pi, np.pi)
    new_signal = signal * np.exp(1j * theta)
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
    alpha = 0.016#0.027 #0.132 immediate phase correction based on current error
    beta = 0.00563#0.00286 #0.00932  tracks accumalated phase error
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
    print(f' converged frequency offset: {freq_log[-1]}')

    plt.plot(freq_log,'.-')
    plt.title('freq converge')
    plt.show()
    t = np.arange(len(qpsk_wave)) / fs
    return qpsk_wave * np.exp(-1j * 2 * np.pi * freq_log[-1] * t)
    

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

    #Interpolate signal and match filter
    

    strt = time.time()
    # Binary search for frequency offset
    freq_found = binary_search(rx_signal, marker_wave, min_freq, max_freq)

    print(f"Total time for binary search: {time.time() - strt} s.")

    print(f"Binary Search CAF: {freq_found}")

    t = np.arange(len(rx_signal)) / fs
    return rx_signal * np.exp(-1j * 2 * np.pi * freq_found * t) 

# Notes: No need for coarse or fine as CAF will correct freq in this test

def cross_corr(signal):
    # Pass RRC filter through RRC signal
    _, h = rrc_filter(0.4, 301, 1/symb_rate, fs)
    delay = (301 - 1) // 2 
    #rc_signal = np.convolve(signal, h, mode = 'full')    
    #rc_signal = rc_signal[delay: delay + len(signal)]

    # Start and end markers
    sig_gen = SigGen(0, 1.0, fs, symb_rate)
    start_sequence = sig_gen.start_sequence
    _, start_waveform = sig_gen.generate_qpsk(start_sequence)
    #start_filter = np.convolve(start_waveform, h, mode = 'full')    
    #start_filter = start_filter[delay: delay + len(start_waveform)]

    end_sequence = sig_gen.end_sequence
    _, end_waveform = sig_gen.generate_qpsk(end_sequence)
    #end_filter = np.convolve(end_waveform, h, mode = 'full')    
    #end_filter = end_filter[delay: delay + len(end_waveform)]

    start_corr_sig = np.convolve(signal, np.conj(np.flip(start_waveform)), mode = 'same')
    
    plt.figure()
    plt.title('start correlation')
    plt.plot(np.abs(start_corr_sig))
    end_corr_signal = np.convolve(signal, np.conj(np.flip(end_waveform)), mode = 'same')

    plt.figure()
    plt.plot(np.abs(end_corr_signal))
    plt.title('end correlation')
    plt.show()
    #get the index
    start = np.argmax(np.abs(start_corr_sig)) - int((32) * (fs/symb_rate)) # If the preamble is 32 bits long, its 16 symbols, symbols * samples/symbol = samples
    end = np.argmax(np.abs(end_corr_signal)) + int((8) * (fs/symb_rate))
    print(f'start: {start}, end: {end}')


    signal = signal[start:end]

    return signal

def main():
    #Generate QPSK at Carrier Frequency
    sig_gen = SigGen(freq=900e6, amp=1,sample_rate=fs, symbol_rate=symb_rate)
    bits = sig_gen.message_to_bits('hello there' * 3)
    t, qpsk_wave = sig_gen.generate_qpsk(bits)    

    # Integer time delay
    #qpsk_wave, t = integer_delay(qpsk_wave, 100)
    
    # Set frequency offset
    qpsk_wave = qpsk_wave * np.exp(1j* 2 * np.pi * freq_offset * t) # shifts down by freq offset
    
    # Uniformly random phase offset
    #qpsk_wave = phase_offset(qpsk_wave)

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
    signal_ready = np.convolve(final_fixed_sig, h, mode = 'full')    
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
    
    bits = bits[128:-32]
    bits = ''.join(str(bit) for bit in bits)
    # convert the bits into a string
    
    decoded_string = ''.join(chr(int(bits[i*8:i*8+8],2)) for i in range(len(bits)//8))
    print(f"The decoded message = {decoded_string}")

if __name__ == "__main__":
    main()