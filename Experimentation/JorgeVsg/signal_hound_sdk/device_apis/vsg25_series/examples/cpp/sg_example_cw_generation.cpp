#include "sg_api.h"

#include <cstdio>
#include <Windows.h>

void sg_example_cw_generation()
{
    // Open device, get handle, check open result
    int handle;
    sgStatus status = sgOpenDevice(&handle);
    if(status < sgNoError) {
        printf("Error: %s\n", sgGetStatusString(status));
        return;
    }

    // Configure generator
    const double freq = 1.0e9; // Hz
    const double level = -10.0; // dBm

    sgSetFrequencyAmplitude(handle, freq, level);

    // Output CW
    sgSetCW(handle);

    // Will transmit until you close the device or abort
    Sleep(5000);

    // Stop waveform
    sgRFOff(handle);

    // Done with device
    sgCloseDevice(handle);
}
