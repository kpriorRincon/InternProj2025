# -*- coding: utf-8 -*-

# This example generates a pulse signal using the custom I/Q interface.

from sgdevice.sg_api import *
from time import sleep

def custom_iq_pulse():
    # Open device
    device = sg_open_device()["handle"]

    # Configure generator
    freq = 1.0e9 # Hz
    level = -20.0 # dBm

    sg_set_frequency_amplitude(device, freq, level)

    # Output pulse, two I/Q values of {1,0} followed by {0,0}
    i = numpy.zeros(2).astype(numpy.double)
    q = numpy.zeros(2).astype(numpy.double)
    i[0] = 1

    sg_set_custom_iq(device, 200e3, i, q, 2, 8)

    # Will transmit until you close the device or abort
    sleep(5)

    # Done with device
    sg_close_device(device)

if __name__ == "__main__":
    custom_iq_pulse()
