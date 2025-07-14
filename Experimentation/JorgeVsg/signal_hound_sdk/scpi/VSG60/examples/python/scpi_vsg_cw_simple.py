# This example demonstrates
# 1. Using PyVISA (https://pyvisa.readthedocs.io/) to connect to the VSG60 software
# 2. Configuring the software for a CW output
# 3. Waiting for the *OPC bit to be set in the status register

import pyvisa
from time import sleep

def wait_for_opc(inst):
    inst.write("*OPC")
    while True:
        esr = int(inst.query("*ESR?"))
        if esr:
            break
        sleep(16e-3)

def scpi_vsg_cw_simple():
    # Get the VISA resource manager
    rm = pyvisa.ResourceManager()

    # Open a session to the Spike software, Spike must be running at this point
    inst = rm.open_resource('TCPIP::localhost::5024::SOCKET')

    # For SOCKET programming, we want to tell VISA to use a terminating character
    #   to end a read and write operation.
    inst.read_termination = '\n'
    inst.write_termination = '\n'

    inst.write("OUTPUT ON")
    inst.write("OUTPUT:MOD OFF")
    inst.write("FREQ 1GHz")
    inst.write("POW -30")

    wait_for_opc(inst)

    # Do your measurement here

    # Done
    inst.close()

if __name__ == "__main__":
    scpi_vsg_cw_simple()
