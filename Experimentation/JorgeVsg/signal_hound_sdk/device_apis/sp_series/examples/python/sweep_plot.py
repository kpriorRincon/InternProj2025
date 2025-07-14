# -*- coding: utf-8 -*-

# This example configures the receiver for a basic sweep and
# plots the sweep. The x-axis frequency is derived from the start_freq
# and bin_size values returned after configuring the device.

from spdevice.sp_api import *

import matplotlib.pyplot as plt
import seaborn as sns; sns.set() # styling

def sweep():
    # Open device
    handle = sp_open_device()["device"]

    # Configure device
    sp_set_sweep_center_span(handle, 1e9, 100e6)
    sp_set_ref_level(handle, -30.0)
    sp_set_sweep_coupling(handle, 10e3, 10e3, 0.001)
    sp_set_sweep_detector(handle, SpDetector.spDetectorMinMax, SpVideoUnits.spVideoLog)
    sp_set_sweep_scale(handle, SpScale.spScaleLog)
    sp_set_sweep_window(handle, SpWindowType.spWindowFlatTop)
    sp_set_sweep_spur_reject(handle, False)

    # Initialize
    sp_configure(handle, SpMode.spModeSweeping)
    query = sp_get_sweep_parameters(handle)
    actual_start_freq = query["actual_start_freq"]
    bin_size = query["bin_size"]
    sweep_size = query["sweep_size"]

    # Get sweep
    sweep = sp_get_sweep(handle)

    # Device no longer needed, close it
    sp_close_device(handle)

    # Plot
    freqs = [actual_start_freq + i * bin_size for i in range(sweep_size)]
    plt.plot(freqs, sweep["sweep_max"])
    plt.show()

if __name__ == "__main__":
    sweep()
