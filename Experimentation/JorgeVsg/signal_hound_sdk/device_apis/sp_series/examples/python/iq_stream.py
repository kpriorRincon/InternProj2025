# -*- coding: utf-8 -*-

# This example tests the throughput of the I/Q acquisition mode
# of the receiver. Adjust the DECIMATION parameter to decimate
# the I/Q data stream. DECIMATION must be a power of two.
# You should see samples rates of 50 MS/sec / DECIMATION

from spdevice.sp_api import *
import datetime

SAMPLES_PER_CAPTURE = 262144
NUM_CAPTURES = 1000
DECIMATION = 1
BANDWIDTH = 20.0e6 / DECIMATION

def stream_iq():
    # Open device
    handle = sp_open_device()["device"]

    # Configure device
    sp_set_IQ_center_freq(handle, 10e9)
    sp_set_IQ_sample_rate(handle, DECIMATION)
    sp_set_IQ_bandwidth(handle, BANDWIDTH)
    sp_set_IQ_software_filter(handle, False)

    # Initialize
    sp_configure(handle, SpMode.spModeIQStreaming)

    # Stream I/Q
    print ("Streaming...")
    sample_count = 0
    start_time = datetime.datetime.now()
    for i in range(NUM_CAPTURES):
        iq_buf = sp_get_IQ(handle, SAMPLES_PER_CAPTURE, 0, False)["iq_buf"]
        sample_count += SAMPLES_PER_CAPTURE

    # Print stats
    time_diff = (datetime.datetime.now() - start_time).total_seconds()
    print (f"\nCaptured {sample_count} samples @ {sample_count / time_diff / 1e6} megasamples/sec")

    # Close device
    sp_close_device(handle)


if __name__ == "__main__":
    stream_iq()
