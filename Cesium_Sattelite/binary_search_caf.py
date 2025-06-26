import numpy as np
import matplotlib.pyplot as plt
import time
from Sig_Gen import SigGen, rrc_filter
freq_offset = 81
fs = 2.88e6
symb_rate = 1e6/20
max_freq = 200
min_freq = -200
start_sequence = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1,
 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1,
 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1]
print(len(start_sequence))

def mixing(signal, f):
    t = np.arange(len(signal)) / fs
    return signal * np.exp(1j * 2 * np.pi * f * t)

def correlate(signal, match_filter):
    # Convolve filter with signal and extract highest correlation
    energy_map = np.convolve(signal, np.conj(np.flip(match_filter)), mode = 'same')
    highest_correlation = np.max(np.abs(energy_map))    

    return highest_correlation

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
    print(f"Energy at {l_freq} Hz: {l_energy}. Energy at {r_freq} Hz: {r_energy}.")
    if r_energy > l_energy:
        l_freq += (r_freq - l_freq) // 2
        freq = binary_search(rx_signal, match_filter, l_freq, r_freq)
    
    elif r_energy < l_energy:
        #print("Should be here")
        r_freq -= (r_freq - l_freq) // 2
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
    rrc_start_filter = np.convolve(marker_wave, h, mode = 'full')    
    rrc_start_filter = rrc_start_filter[delay: delay + len(marker_wave)]

    # Binary search for frequency offset
    freq_found = binary_search(rx_signal, rrc_start_filter, min_freq, max_freq)

    return freq_found

# Notes: No need for coarse or fine as CAF will correct freq in this test

def main():
    #Generate QPSK at Carrier Frequency
    sig_gen = SigGen(freq=900e6, amp=1,sample_rate=fs, symbol_rate=symb_rate)
    bits = sig_gen.message_to_bits('hello there' * 3)
    t, qpsk_wave = sig_gen.generate_qpsk(bits)    

    # Set frequency offset
    qpsk_wave = qpsk_wave * np.exp(1j* 2 * np.pi * freq_offset * t) # shifts down by freq offset
    
    #Tune down to baseband
    tuned_sig = qpsk_wave * np.exp(-1j * 2 * np.pi * sig_gen.freq * t)

    # Run CAF and return frequency offset found with highest correlation
    freq_off_found = cross_corr_caf(tuned_sig)
    print(freq_off_found)

    #Down convert with offset
    signal_ready = tuned_sig * np.exp(-1j * 2 * np.pi * freq_off_found * t)

    # Pass through RRC filter

    # Decimate
    """
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
    
    bits = bits[32:-32]
    bits = ''.join(str(bit) for bit in bits)
    # convert the bits into a string
    
    decoded_string = ''.join(chr(int(bits[i*8:i*8+8],2)) for i in range(len(bits)//8))
    print(f"The decoded message = {decoded_string}")
    """

if __name__ == "__main__":
    main()