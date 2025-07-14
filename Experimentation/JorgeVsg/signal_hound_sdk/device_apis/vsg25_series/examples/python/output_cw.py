# -*- coding: utf-8 -*-

# This example generates a basic CW signal.

from sgdevice.sg_api import *
from time import sleep

def output_cw():
    # Open device
    device = sg_open_device()["handle"]

    # Configure the device to output a -20 dBm CW signal at 1GHz
    sg_set_frequency_amplitude(device, 1.0e9, -20.0)
    sg_set_cw(device)

    # Will transmit until you close the device or abort
    sleep(5)

    # Done with device
    sg_close_device(device)

if __name__ == "__main__":
    output_cw()
