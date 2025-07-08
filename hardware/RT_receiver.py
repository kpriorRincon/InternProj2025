#from rtlsdr import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import receive_processing as rp
import time
import transmit_processing as tp
from RT_channel_correction import *
from config import *
from rtlsdr import *
import queue
import threading

#transmit_obj = tp.transmit_processing(int(SAMPLE_RATE/SYMB_RATE), SAMPLE_RATE)
#match_start, match_end = transmit_obj.modulated_markers(BETA, NUMTAPS) 

threshold = 8
N = SPS * 1024
BOUNDS = 100
QUEUE_SIZE = 100

cut_sigs = None # if cut_sigs has stuff, loop through, if first half, 


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
    distance = len(marker_filter) * 2
    # normalize the samples
    # samples = (samples - np.min(samples)) / (np.max(np.abs(samples)) - np.min(samples))
    strt_t = time.time()
    coarse_fixed = coarse_freq_recovery(samples)
    #print(f"Time for coarse freq: {time.time() - strt_t}")
    # default returns 
    start = 0
    end = 0
    detected = False
    

    strt_t = time.time()
    # find the correlated signal
    # start cor
    cor_start = np.abs(signal.fftconvolve(coarse_fixed, np.conj(np.flip(marker_filter)), mode='same'))
    # end cor
    cor_end = np.abs(signal.fftconvolve(coarse_fixed, np.conj(np.flip(end_filter)), mode='same'))
    

    start_peaks = signal.find_peaks(cor_start, distance=distance, height=50)[0]
    end_peaks = signal.find_peaks(cor_end, distance=distance, height=50)[0]

    #print(f"Start peaks {start_peaks}")
    #print(f"End peaks {end_peaks}")
    # get start and end indices
    


    """
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
    """
    # Add extra samples around data for channel correction
    # Edge cases: if start or end are close to bounds of buffer
    #   Have check where if goes below zero or above max, set them to that
    start_peaks -= int(len(marker_filter) / 2) + BOUNDS
    end_peaks += int(len(marker_filter) / 2) + BOUNDS

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
    
    #print(f"Signal pairs: {sig_pairs}")
    #print(f"Cut pairs: {cut_peaks}")

    # These contain pairs of indeces (x1, x2) or (None, x2) or (x1, None)
    signals_found = []
    signals_cut = []
    for i, pairs in enumerate(cut_peaks):
        if pairs[0]: # Cut signal is first half signal (end of buffer)
            signals_cut.append(('f', coarse_fixed[pairs[0]:])) # this one is good
        else: # Cut signal is second half (beginning of buffer) so call prev_cut signal
            for j, old_pair in enumerate(prev_cut):
                if old_pair[0] == 'f':
                    fixed_signal = np.concatenate((old_pair[1], coarse_fixed[:pairs[1]]))
                    signals_found.append(fixed_signal)
                    cut_peaks.pop(i)
                    prev_cut.pop(j)
                    # remove pair from cut_peaks
            # Second half will never be used in future iter

    for pairs in sig_pairs:
        signals_found.append(coarse_fixed[pairs[0]:pairs[1]])
    #print(f"Finding peak time: {time.time() - strt_t}")

    messages = []

    for signal in signals_found:
        strt_t = time.time()
        messages.append(channel_handler(signal))
        #print(f"time for one channel handler: {time.time() - strt_t}")
        #input()

    return messages, signals_cut

# messages is list of decoded messages
# signals_cut is list of tuples (f or first half, s for second half, cut signal)
iq_queue = queue.Queue(maxsize=QUEUE_SIZE)
FORMAT = np.complex64

def callback(samples, rtlsdr_obj):
    try:
        iq_queue.put_nowait(samples.copy())
    except queue.Full:
        print("WARNING: Dropped a block!")

def callback_d(samples, rtlsdr_obj):
    global cut_sigs
    _, cut_sigs = detector(samples, cut_sigs)

def writer_thread():
    with open('iq_dump.bin', 'ab') as f:
        while True:
            data = iq_queue.get()
            if data is None:
                break  # Exit signal
            f.write(data.astype(FORMAT).tobytes())

def run_RTL_SDR():
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

    # create transmit object and get the start and end markers

    # Test SDR connection before main loop
    try:
        test_samples = sdr.read_samples(1024)
        print(f"SDR test successful: read {len(test_samples)} samples")
    except Exception as e:
        print(f"SDR test failed: {e}")
        sdr.close()
        exit()

    """
    while True:
        data = sdr.read_samples(N)
        with open('iq_dump.bin', 'ab') as f:
            f.write(data.astype(FORMAT).tobytes())
    """
    #find end idx in cut pairs, if sec half, find start idx in cut pairs

    sdr.read_samples_async(callback_d, N)
    #messages, cut_sigs = detector(samples, cut_sigs)

    #sdr.read_samples_async(callback, 1024)


    sdr.close()


def main():
    
    #with open('iq_dump.bin', 'wb'):
    #    pass  # just open and close to truncate file

    #writer = threading.Thread(target=writer_thread, daemon=True)
    #writer.start()
    run_RTL_SDR()
    return
    
    raw_data = np.fromfile("iq_dump.bin", dtype=np.complex64)
    print("Length of data in file: ", len(raw_data))


    i = 0
    cut_sigs = None # if cut_sigs has stuff, loop through, if first half, 
    #find end idx in cut pairs, if sec half, find start idx in cut pairs
    while i + N < len(raw_data) + 1:
        print(f"\nWindow #{(i // N) + 1}")
        samples = raw_data[i: i + N]
        str_t = time.time()
        messages, cut_sigs = detector(samples, cut_sigs)
        i += N
        print(f"Total T: {time.time() - str_t}")
        input()
    #while True: # Real time reading
    #    messages, cut_signals = detector(raw_data, cut)


    #print("Raw data loaded from file:\n", raw_data)
    #bits_string, decoded_message = channel_handler(raw_data)


if __name__ == "__main__":
    main()


# Notes:
# Not even 
# Length of signal ~ 3880
# Losing around 1574 - 1644 samples