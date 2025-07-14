#include <cstdio>
#include <cstdlib>

#include "pn_api.h"

// The serial port of the PN400 can be found as follows
//   - Windows: Device Manager > Ports (COM & LPT)
//   - Linux: ls /dev | grep ttyACM
#define SERIAL_PORT 0

// Configure the VCO ports on a PN400
void pn_example_vco()
{
    // Open the PN400 at the specified serial port
    int handle = -1;
    PnStatus status = pnOpenDevice(SERIAL_PORT, &handle);

    if(status != pnNoError) {
        printf("Unable to open device\n");
        exit(-1);
    } 

    printf("Device opened successfully\n");

    // Get serial number and firmware
    int serial = 0, firmware = 0;
    pnGetSerialAndFirmware(handle, &serial, &firmware);

    // Set V Supply
    pnSetVcc(handle, 5); // volts

    // Set V Tune
    pnSetVtune(handle, 12); // volts

    // Enable VCO outputs
    pnEnableOutput(handle, true);

    // Finished
    pnCloseDevice(handle);
}
