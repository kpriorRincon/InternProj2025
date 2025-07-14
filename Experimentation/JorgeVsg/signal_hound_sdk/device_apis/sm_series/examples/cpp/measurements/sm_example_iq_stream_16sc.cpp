#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sm_api.h"

// This example is identical to sm_example_iq_stream() except it sets the data type to
//  16-bit complex shorts, and performs the conversion to 32-bit floats

void sm_example_iq_stream_16sc()
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
    smSetRefLevel(handle, -20.0);
    smSetIQSampleRate(handle, 2); // 50 / 2 = 25MS/s IQ 
    smSetIQBandwidth(handle, smTrue, 20.0e6); // 20MHz of bandwidth
    smSetIQDataType(handle, smDataType16sc);

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

    float iqCorrection = 0.0;
    smGetIQCorrection(handle, &iqCorrection);

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
        status = smGetIQ(handle, &iqBuf16sc[0], bufLen, 0, 0, 0, smFalse, 0, 0);

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
    smCloseDevice(handle);
}