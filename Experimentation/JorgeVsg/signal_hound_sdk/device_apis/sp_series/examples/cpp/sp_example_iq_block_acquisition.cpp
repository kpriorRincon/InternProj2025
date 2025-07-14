#include <cstdio>
#include <cstdlib>

#include "sp_api.h"

// Open a device
// Configure and perform a small IQ block acquisition

void sp_example_iq_block_acquisition()
{
    int handle = -1;
    SpStatus status = spNoError;

    // Open device
    status = spOpenDevice(&handle);

    if(status != spNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // Configure the receiver for IQ acquisition
    spSetIQCenterFreq(handle, 900.0e6); // 900MHz
    spSetIQSampleRate(handle, 2); // 50 / 2 = 25MS/s IQ 
    spSetIQBandwidth(handle, 20.0e6); // 20MHz of bandwidth

    // Initialize the receiver with the above settings
    status = spConfigure(handle, spModeIQStreaming);
    if(status != spNoError) {
        printf("Unable to configure device\n");
        printf("%s\n", spGetErrorString(status));
        spCloseDevice(handle);
        exit(-1);
    }

    // Query the receiver IQ stream characteristics
    // Should match what we set earlier
    double actualSampleRate, actualBandwidth;
    spGetIQParameters(handle, &actualSampleRate, &actualBandwidth);

    // Allocate memory for complex sample, IQ pairs interleaved
    int bufLen = 16384;
    float *iqBuf = new float[bufLen * 2]; 

    // Acquire single block of IQ data
    // Set null all parameters we don't care about
    // There is a small filter ramp up time after calling spConfigure
    // Call the getIQ function twice to flush this
    spGetIQ(handle, iqBuf, bufLen, 0, 0, 0, spTrue, 0, 0);
    spGetIQ(handle, iqBuf, bufLen, 0, 0, 0, spTrue, 0, 0);

    // Finished
    spCloseDevice(handle);

    // Clean up
    delete [] iqBuf;
}
