# -*- coding: utf-8 -*-

# This example generates a signal.

from tgdevice.tg_api import *
from time import sleep

def generate_signal():
    device = 0

    # Open device
    tg_open_device(device)

    # Print serial number
    serial = tg_get_serial_number(device)['serial']
    print (f'Serial Number: {serial}')

    # Configure signal
    tg_set_freq_amp(device, 1.0e9, -15)

    # # Will transmit until you close the device or abort
    # sleep(5);

    # Close device
    tg_close_device(device);

if __name__ == "__main__":
    generate_signal()
