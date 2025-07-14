# -*- coding: utf-8 -*-

# This example generates a multi-tone signal.

from sgdevice.sg_api import *
from time import sleep

def output_multitone():
    # Open device
    device = sg_open_device()["handle"]

    # Configure a 16 tone multi-tone signal at 2.4GHz
    # The tones are separated by 10kHz and no notch filter should be applied
    # A parabolic phase is specified for the best dynamic range
    sg_set_frequency_amplitude(device, 2400.0e6, -10.0)
    sg_set_multitone(device, 16, 10.0e3, 0.0, SG_PARABOLIC)

    # Will transmit until you close the device or abort
    sleep(5)

    # Done with device
    sg_close_device(device)

if __name__ == "__main__":
    output_multitone()
