#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sm_api.h"

// This example demonstrates how to interface the API for external triggers
//   during I/Q streaming.
// This example waits a 1 second for an external trigger and if one is seen, then
//   collects 1 second of I/Q data.

void sm_example_iq_ext_trigger()
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
    smSetIQCenterFreq(handle, 2.45e9); // 2.45GHz
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

    // Allocate memory for complex samples, IQ pairs interleaved
    std::vector<float> iqCapture((int)actualSampleRate * 2);
    double trigger = -1;

    printf("Waiting for external trigger\n");

    // Wait N seconds for ext trigger
    const int searchTimeSec = 3; // Number of seconds to wait
    int waitSize = 16384;
    int waitCaptures = searchTimeSec * ((int)actualSampleRate / waitSize + 1);
    while(waitCaptures > 0) {
        smGetIQ(handle, &iqCapture[0], waitSize, &trigger, 1, 0, smFalse, 0, 0);

        if(trigger > 0) {
            printf("Found external trigger !\n");
            break;
        }

        waitCaptures--;
    }

    if(trigger > 0) {
        int iTrigPos = (int)trigger;
        // Found trigger, lets collect up to 1 second of data after the trigger
        // First grab all the samples after the trigger from the previous smGetIQ call
        int remaining = waitSize - iTrigPos;
        int samplesLeft = ((int)iqCapture.size() / 2) - remaining;
        memmove(&iqCapture[0], &iqCapture[iTrigPos * 2], remaining * 2 * sizeof(float));

        // Grab remaining samples
        smGetIQ(handle, &iqCapture[remaining * 2], samplesLeft, 0, 0, 0, smFalse, 0, 0);

        // At this point we have 1 second of IQ data after an external trigger is seen
        // The I/Q data is in the iqCapture buffer, with the first index being the first
        //   sample after the ext trigger.
    } else {
        printf("Did not find external trigger\n");
    }

    // Finished
    smCloseDevice(handle);
}