# -*- coding: utf-8 -*-
  
# This example generates a basic CW signal.
  
from vsgdevice.vsg_api import *
import time
import transmit_processing as transmit_processing
import numpy as np
  
def generate_iq():
    # Open device
    handle = vsg_open_device()["handle"]

    # Configure generator
    freq = 910e6 # Hz
    sample_rate = 2.88e6 # samples per second
    level = -20.0 # dBm
    sps = 20 # Samples per Symbol
    N = 101 # Number of Taps for RRC Filter
    beta = 0.35 # Roll-off Factor for RRC Filter

    vsg_set_frequency(handle, freq)
    vsg_set_level(handle, level);
    vsg_set_sample_rate(handle, sample_rate);

    # Getting IQ Data
    message = input("Enter your message: \n")
    tp = transmit_processing.transmit_processing(sps, sample_rate)
    start = time.time()
    _, data = tp.work(message, beta, N)
    end = time.time()
    total_time = end - start
    print("Modulation Time: ", total_time)

    # Output CW, single I/Q value of {1,0}
    # This is equivalent to calling vsgOutputCW
    iq = np.empty(data.size * 2, dtype=np.float32)
    iq[0::2] = data.real
    iq[1::2] = data.imag
    vsg_repeat_waveform(handle, iq, len(iq));

    # Will transmit until you close the device or abort
    time.sleep(25);

    # Stop waveform
    vsg_abort(handle);

    # Done with device
    vsg_close_device(handle);
  
if __name__ == "__main__":
      generate_iq()                          
