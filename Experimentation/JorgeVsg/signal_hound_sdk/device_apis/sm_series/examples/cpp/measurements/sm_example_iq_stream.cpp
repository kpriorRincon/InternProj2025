/** [iqStreamingExample1] */

// Configure the device for I/Q streaming and stream for a period of time 
// This example assumes a USB 3.0 SM device.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sm_api.h"

void sm_example_iq_stream()
{
    int handle = -1;
    SmStatus status = smNoError;

    // Uncomment this to open a USB SM device
    status = smOpenDevice(&handle);
    // Uncomment this to open a networked SM device with a default network config
    //status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    // Check open status
    if(status != smNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // Configure the receiver for IQ acquisition
    smSetRefLevel(handle, -20.0); // -20 dBm reference level
    smSetIQCenterFreq(handle, 900.0e6); // 900MHz center frequency
    smSetIQBaseSampleRate(handle, smIQStreamSampleRateNative); // Use native 50MS/s base sample rate.
    smSetIQSampleRate(handle, 2); // 50 / 2 = 25MS/s IQ 
    smSetIQBandwidth(handle, smTrue, 20.0e6); // 20MHz of bandwidth
    smSetIQDataType(handle, smDataType32fc);

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
    std::vector<float> iqBuf(bufLen * 2);

    // Let's acquire 5 second worth of data
    int samplesNeeded = 5 * (int)actualSampleRate;

    while(samplesNeeded > 0) {
        // Notice the purge parameter is set to false, so that each time
        //  the get IQ function is called, the next contiguous block of data
        //  is returned.
        smGetIQ(handle, &iqBuf[0], bufLen, 0, 0, 0, smFalse, 0, 0);

        // Process/store data here
        // Data is interleaved 32-bit complex values

        // Need bufLen less samples
        samplesNeeded -= bufLen;
    }

    // Finished
    smCloseDevice(handle);
}

/** [iqStreamingExample1] */