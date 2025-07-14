#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sp_api.h"

// This example is identical to sp_example_iq_stream() except it sets the data type to
//  16-bit complex shorts, and performs the conversion to 32-bit floats

void sp_example_iq_stream_16sc()
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
    spSetRefLevel(handle, -20.0);
    spSetIQSampleRate(handle, 2); // 50 / 2 = 25MS/s IQ 
    spSetIQBandwidth(handle, 20.0e6); // 20MHz of bandwidth
    spSetIQDataType(handle, spDataType16sc);

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

    float iqCorrection = 0.0;
    spGetIQCorrection(handle, &iqCorrection);

    // Allocate memory for complex sample, IQ pairs interleaved
    int bufLen = 16384;
    std::vector<short> iqBuf16sc(bufLen * 2);
    std::vector<float> iqBuf32fc(bufLen * 2);

    // Let's acquire 5 second worth of data
    int samplesNeeded = 5 * (int)actualSampleRate;

    while(samplesNeeded > 0) {
        // Notice the purge parameter is set to false, so that each time
        //  the get IQ function is called, the next contiguous block of data
        //  is returned.
        status = spGetIQ(handle, &iqBuf16sc[0], bufLen, 0, 0, 0, spFalse, 0, 0);

        // Convert to float, I/Q data is interleaved, apply the same logic to all re/im samples.
        for(int i = 0; i < iqBuf16sc.size(); i++) {
            // Convert to floats in the range [-1.0, 1.0]
            iqBuf32fc[i] = (((float)iqBuf16sc[i]) / 32768.0);
            // Apply correction.
            iqBuf32fc[i] *= iqCorrection;
        }

        // Process data here

        // Need bufLen less samples
        samplesNeeded -= bufLen;
    }

    // Finished
    spCloseDevice(handle);
}