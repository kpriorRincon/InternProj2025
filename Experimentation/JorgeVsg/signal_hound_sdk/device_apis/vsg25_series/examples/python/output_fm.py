# -*- coding: utf-8 -*-

# This example generates an FM signal.

from sgdevice.sg_api import *
from time import sleep

def output_fm():
    # Open device
    device = sg_open_device()["handle"]

    # Configure the device to output a 12kHz sinusoidal FM waveform with a deviation of 100kHz at 91.3MHz
    sg_set_frequency_amplitude(device, 91.3e6, -10.0)
    sg_set_fm(device, 12.0e3, 100.0e3, SG_SHAPE_SINE)

    # Will transmit until you close the device or abort
    sleep(5)

    # Done with device
    sg_close_device(device)

if __name__ == "__main__":
    output_fm()
