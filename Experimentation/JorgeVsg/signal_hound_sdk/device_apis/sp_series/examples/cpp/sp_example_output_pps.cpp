#include "sp_api.h"

#include <cstdio>

// Configure the GPIO port to output the internal PPS
void sp_example_output_pps()
{
    int handle = -1;
    SpStatus status = spOpenDevice(&handle);
    if(status != spNoError) {
        printf("Unable to open device\n");
        return;
    }

    // This takes effect immediately
    spSetGPIOPort(handle, SpGPIOFunctionPPSOut); 

    // Assuming the GPS antenna is attached to the device, once the device
    // achieves lock, the PPS should be visible on the GPIO port.
    // This will persist through all measurement taking.

    // Perform measurements here

    spCloseDevice(handle);
}
