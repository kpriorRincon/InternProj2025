# -*- coding: utf-8 -*-

# This example configures the receiver for a basic sweep and
# plots the sweep. The x-axis frequency is derived from the start_freq
# and bin_size values returned after configuring the device.

from bbdevice.bb_api import *

import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # styling

def sweep():
    # Open device
    handle = bb_open_device()["handle"]

    # Configure device
    bb_configure_center_span(handle, 1.0e9, 100.0e6)
    bb_configure_ref_level(handle, -30.0)
    bb_configure_gain_atten(handle, BB_AUTO_GAIN, BB_AUTO_ATTEN)
    bb_configure_sweep_coupling(handle, 10.0e3, 10.0e3, 0.001, BB_RBW_SHAPE_FLATTOP, BB_NO_SPUR_REJECT)
    bb_configure_acquisition(handle, BB_MIN_AND_MAX, BB_LOG_SCALE)
    bb_configure_proc_units(handle, BB_POWER)

    # Initialize
    bb_initiate(handle, BB_SWEEPING, 0)
    query = bb_query_trace_info(handle)
    trace_len = query["trace_len"]
    start = query["start"]
    bin_size = query["bin_size"]

    # Get sweep
    trace_max = bb_fetch_trace_32f(handle, trace_len)["trace_max"]

    # Device no longer needed, close it
    bb_close_device(handle)

    # Plot
    freqs = [start + i * bin_size for i in range(trace_len)]
    plt.plot(freqs, trace_max)
    plt.show()

if __name__ == "__main__":
    sweep()
