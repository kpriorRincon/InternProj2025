from rtlsdr import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.signal as signal
import Detector as d
import receive_processing as rp
import time
import transmit_processing as tp
from channel_correction import *
from config import *

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
    beta = BETA # excess bandwidth factor for the filter
    num_taps = NUMTAPS      # number of taps for the filter
    symbol_rate = SYMB_RATE # symbol rate calculated from sample rate and samples per symbol

    # create transmit object and get the start and end markers
    transmit_obj = tp.transmit_processing(sps, sdr.sample_rate)
    match_start, match_end = transmit_obj.modulated_markers(beta, num_taps) 

    # detector object
    detect_obj = d.Detector(N, 1 / symbol_rate, beta, sdr.sample_rate, sps=sps)

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
    while detected == False:
        count += 1  # increment cycle count
        
        # read samples from RTL-SDR
        samples = None
        samples = sdr.read_samples(N)

        # run detection
        detected, corrected_data = detect_obj.detector(samples, match_start=match_start, match_end=match_end)

    # take signal from the samples
    print(f"Signal found after {count} cycles")

    # begin signal processing
    print("Processing data...")
    bits_string, decoded_message = channel_handler(corrected_data)

    # close sdr
    sdr.close()

    # return the 
    return bits_string, decoded_message
