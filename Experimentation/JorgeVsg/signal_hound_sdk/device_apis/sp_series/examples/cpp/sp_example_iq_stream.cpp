/** [iqStreamingExample1] */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sp_api.h"

// Configure the device for I/Q streaming and stream for a period of time.

void sp_example_iq_stream()
{
    int handle = -1;
    SpStatus status = spNoError;

    // Open device
    status = spOpenDevice(&handle);

    // Check open status
    if(status != spNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // Configure the receiver for IQ acquisition
    spSetRefLevel(handle, -20.0); // -20 dBm reference level
    spSetIQCenterFreq(handle, 900.0e6); // 900MHz center frequency
    spSetIQSampleRate(handle, 2); // 50 / 2 = 25MS/s IQ 
    spSetIQBandwidth(handle, 20.0e6); // 20MHz of bandwidth
    spSetIQDataType(handle, spDataType32fc);

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
    std::vector<float> iqBuf(bufLen * 2);

    // Let's acquire 5 second worth of data
    int samplesNeeded = 5 * (int)actualSampleRate;

    while(samplesNeeded > 0) {
        // Notice the purge parameter is set to false, so that each time
        //  the get IQ function is called, the next contiguous block of data
        //  is returned.
        spGetIQ(handle, &iqBuf[0], bufLen, 0, 0, 0, spFalse, 0, 0);

        // Process/store data here
        // Data is interleaved 32-bit complex values

        // Need bufLen less samples
        samplesNeeded -= bufLen;
    }

    // Finished
    spCloseDevice(handle);
}

/** [iqStreamingExample1] */
