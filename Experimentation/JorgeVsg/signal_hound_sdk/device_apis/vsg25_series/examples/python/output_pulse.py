# -*- coding: utf-8 -*-

# This example generates a pulse signal.

from sgdevice.sg_api import *
from time import sleep

def output_pulse():
    # Open device
    device = sg_open_device()["handle"]

    # Configure a 40us pulse with a 50% duty cycle in the ISM band at 915MHz
    sg_set_frequency_amplitude(device, 915.0e6, 10.0)
    sg_set_pulse(device, 80.0e-6, 40.0e-6)

    # It might be important to determine the exact
    # pulse width and period used by the device
    result = sg_query_pulse(device)
    period = result["period"]
    width = result["width"]

    # Will transmit until you close the device or abort
    sleep(5)

    # Done with device
    sg_close_device(device)

if __name__ == "__main__":
    output_pulse()
