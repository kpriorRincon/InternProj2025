# -*- coding: utf-8 -*-

# This example configures the VCO outputs of the PN400

from pndevice.sp_api import *

# The serial port of the PN400 can be found as follows
#   - Windows: Device Manager > Ports (COM & LPT)
#   - Linux: ls /dev | grep ttyACM
SERIAL_PORT = 0

def vco():
    # Open device
    handle = pn_open_device(SERIAL_PORT)["device"]

    # Set V Supply
    pn_set_vcc(handle, 5); # volts

    # Set V Tune
    pn_set_vtune(handle, 12); # volts

    # Enable VCO outputs
    pn_enable_output(handle, True);

    # Finished
    pn_close_device(handle);

if __name__ == "__main__":
    vco()
