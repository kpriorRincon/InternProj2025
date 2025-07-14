#include <cstdio>
#include <cstdlib>

#include "sm_api.h"

// Open a device
// Configure and perform a small IQ block acquisition

void sm_example_iq_block_acquisition()
{
    int handle = -1;
    SmStatus status = smNoError;

    // Uncomment this to open a USB SM device
    status = smOpenDevice(&handle);
    // Uncomment this to open a networked SM device with a default network config
    //status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    if(status != smNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // Configure the receiver for IQ acquisition
    smSetIQCenterFreq(handle, 900.0e6); // 900MHz
    smSetIQSampleRate(handle, 2); // 50 / 2 = 25MS/s IQ 
    smSetIQBandwidth(handle, smTrue, 20.0e6); // 20MHz of bandwidth

    // Initialize the receiver with the above settings
    status = smConfigure(handle, smModeIQ);
    if(status != smNoError) {
        printf("Unable to configure device\n");
        printf("%s\n", smGetErrorString(status));
        smCloseDevice(handle);
        exit(-1);
    }

    // Query the receiver IQ stream characteristics
    // Should match what we set earlier
    double actualSampleRate, actualBandwidth;
    smGetIQParameters(handle, &actualSampleRate, &actualBandwidth);

    // Allocate memory for complex sample, IQ pairs interleaved
    int bufLen = 16384;
    float *iqBuf = new float[bufLen * 2]; 

    // Acquire single block of IQ data
    // Set null all parameters we don't care about
    // There is a small filter ramp up time after calling smConfigure
    // Call the getIQ function twice to flush this
    smGetIQ(handle, iqBuf, bufLen, 0, 0, 0, smTrue, 0, 0);
    smGetIQ(handle, iqBuf, bufLen, 0, 0, 0, smTrue, 0, 0);

    // Finished
    smCloseDevice(handle);

    // Clean up
    delete [] iqBuf;

}