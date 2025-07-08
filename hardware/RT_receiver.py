#from rtlsdr import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import receive_processing as rp
import time
import transmit_processing as tp
from channel_correction import *
from config import *

transmit_obj = tp.transmit_processing(int(SAMPLE_RATE/SYMB_RATE), SAMPLE_RATE)
match_start, match_end = transmit_obj.modulated_markers(BETA, NUMTAPS) 

threshold = 8
N = SPS * 1024

def detector(samples, prev_cut):
    import scipy.signal as signal

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
    distance = len(match_start) * 2
    print("Length of samples: ", len(samples))
    # normalize the samples
    # samples = (samples - np.min(samples)) / (np.max(np.abs(samples)) - np.min(samples))
    coarse_fixed = coarse_freq_recovery(samples)
    # default returns 
    start = 0
    end = 0
    detected = False
    
    # find the correlated signal
    # start cor
    cor_start = np.abs(signal.fftconvolve(coarse_fixed, np.conj(np.flip(match_start)), mode='same'))
    # end cor
    cor_end = np.abs(signal.fftconvolve(coarse_fixed, np.conj(np.flip(match_end)), mode='same'))
    

    start_peaks = signal.find_peaks(cor_start, distance=distance, height=30)[0]
    end_peaks = signal.find_peaks(cor_end, distance=distance, height=30)[0]

    print(f"Start peaks {start_peaks}")
    print(f"End peaks {end_peaks}")
    # get start and end indices
    



    plt.subplot(2, 1, 1)
    plt.title('Correlation of the Matched Filters')
    plt.plot(np.abs(cor_start), label='start')
    plt.grid()
    plt.legend()
    plt.axhline(y = threshold, linestyle = '--', color = 'g')
    peak_vals = [cor_start[i] for i in start_peaks]
    plt.scatter(start_peaks, peak_vals, s = 99, c = 'r', marker = '.')
            
    plt.subplot(2, 1, 2)
    plt.plot(np.abs(cor_end), label='end')
    plt.grid()
    plt.legend()
    plt.axhline(y = threshold, linestyle = '--', color = 'g')
    peak_vals = [cor_end[i] for i in end_peaks]

    plt.scatter(end_peaks, peak_vals, s = 99, c = 'r', marker = '.')
    #plt.axvline(x = end, linestyle = '--', color = 'r')
    plt.show()

    start_peaks -= int(len(match_start) / 2)
    end_peaks += int(len(match_start) / 2)

    start_peaks = [idx for idx in start_peaks if idx >= 0]
    end_peaks = [idx for idx in end_peaks if idx <= N]
    sig_pairs = []
    cut_peaks = []

    used_ends = set()
    # start = 2
    # end = 1
    end_idx = 0
    for start in start_peaks:
        while end_idx < len(end_peaks) and end_peaks[end_idx] <= start:
            end_idx += 1

        if end_idx < len(end_peaks):
            end = end_peaks[end_idx]
            sig_pairs.append((start, end))
            used_ends.add(end_idx)
            end_idx += 1
        else:
            cut_peaks.append((start, None))
    
    for i, end in enumerate(end_peaks):
        if i not in used_ends:
            cut_peaks.append((None, end))
    
    print(sig_pairs)
    print(cut_peaks)

    signals_found = []
    signals_cut = []

    for pairs in cut_peaks:
        if pairs[0]:
            signals_cut.append(('f', coarse_fixed[pairs[0]:]))
        else:
            #for i, old_pair in enumerate(prev_cut):
            #if old_pair[0] == 's':
            #        signals_found.append(('f', coarse_fixed[pairs[0]:]))
            signals_cut.append(('s', coarse_fixed[:pairs[1]]))

    for pairs in sig_pairs:
        signals_found.append(coarse_fixed[pairs[0]:pairs[1]])

    
    messages = []
    for signal in signals_found:
        messages.append(channel_handler(signal))

    return messages, signals_cut

# messages is list of decoded messages
# signals_cut is list of tuples (f or first half, s for second half, cut signal)

def run_receiver():
    # configure RTL-SDR
    sdr = RtlSdr()                  # RTL-SDR object
    sdr.sample_rate = SAMPLE_RATE   # sample rate in Hz
    sdr.center_freq = RX_REC_FREQ   # center frequency in Hz
    sdr.freq_correction = PPM       # how much to correct for frequency offset PPM
    sdr.gain = 'auto'               # set gain to auto 

    # sleep to let the SDR settle
    time.sleep(1)

    # settings to run detector
    detected = False        # flag to indicate if the signal is detected
    sps = SPS               # samples per symbol
    N = sps * 1024          # number of samples to read
    beta = EXCESS_BANDWIDTH # excess bandwidth factor for the filter
    num_taps = NUMTAPS      # number of taps for the filter
    symbol_rate = SYMB_RATE # symbol rate calculated from sample rate and samples per symbol

    # create transmit object and get the start and end markers
    transmit_obj = tp.transmit_processing(sps, sdr.sample_rate)
    match_start, match_end = transmit_obj.modulated_markers(beta, num_taps) 

    # Test SDR connection before main loop
    try:
        test_samples = sdr.read_samples(1024)
        print(f"SDR test successful: read {len(test_samples)} samples")
    except Exception as e:
        print(f"SDR test failed: {e}")
        sdr.close()
        exit()

    # run detection
    count = 0   # count cycles until detected
    while True:
        count += 1  # increment cycle count
        
        # read samples from RTL-SDR
        samples = None
        samples = sdr.read_samples(N)

        # run detection
        corrected_data = detector(samples, match_start=match_start, match_end=match_end)

        if corrected_data:
            # do rest of channel correction
            pass
        


def main():
    raw_data = np.fromfile("test_this_john.bin", dtype=np.complex64)
    print("Length of data in file: ", len(raw_data))


    i = 0
    cut_sigs = None # if cut_sigs has stuff, loop through, if first half, 
    #find end idx in cut pairs, if sec half, find start idx in cut pairs
    while i + N < len(raw_data) + 1:
        samples = raw_data[i: i + N]
        messages, cut_sigs = detector(samples, cut_sigs)
        i += N
    #while True: # Real time reading
    #    messages, cut_signals = detector(raw_data, cut)


    #print("Raw data loaded from file:\n", raw_data)
    #bits_string, decoded_message = channel_handler(raw_data)


if __name__ == "__main__":
    main()



