# -*- coding: utf-8 -*-

# This example generates a digitally modulated QPSK signal.

from sgdevice.sg_api import *
from time import sleep

def output_dig_mod():
    # Open device
    device = sg_open_device()["handle"]

    # Configure the device to output a QPSK modulated signal
    # Output the signal at 890 MHz, symbol rate of 3.84 MHz
    # 128 symbols total with root raised cosine filter with
    # roll-off factor of 0.22
    sg_set_frequency_amplitude(device, 890.0e6, -10.0)

    symbols = numpy.zeros(128).astype(c_int)
    # Set symbol values between 0-3

    sg_set_psk(device, 3840.0e3, SG_MOD_QPSK, SG_ROOT_RAISED_COSINE, 0.22, symbols, 128)

    # Will transmit until you close the device or abort
    sleep(5)

    # Done with device
    sg_close_device(device)

if __name__ == "__main__":
    output_dig_mod()
