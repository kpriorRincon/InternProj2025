/** [extTriggerExample1] */

#include "bb_api.h"

#include <complex>
#include <iostream>
#include <vector>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example demonstrates using the BB60 to look for an external trigger.
This example waits for an external trigger, and captures N I/Q samples after
  that trigger. If no trigger arrives this example will loop indefinitely.
*/

const int TRIGGER_SENTINEL = -1;

void bbExampleIQExtTrigger()
{
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << "Issue opening device\n";
        std::cout << bbGetErrorString(status) << "\n";
        return;
    }

    int deviceType;
    bbGetDeviceType(handle, &deviceType);

    // Configure BNC port 2 for rising edge trigger detection
    if(deviceType == BB_DEVICE_BB60D) {
        // BB60D
        bbConfigureIO(handle, BB60D_PORT1_DISABLED, BB60D_PORT2_IN_TRIG_RISING_EDGE);
    } else {
        // BB60C/A devices
        bbConfigureIO(handle, 0, BB60C_PORT2_IN_TRIG_RISING_EDGE);
    }

    // Now configure the measurement parameters

    // We want 32-bit floating point complex values
    bbConfigureIQDataType(handle, bbDataType32fc);
    // Set center frequency to 1GHz
    bbConfigureIQCenter(handle, 1.0e9);
    // Set reference level 
    bbConfigureRefLevel(handle, -20.0);
    // Set a sample rate of 40.0e6 / 2 = 20.0e6 MS/s and bandwidth of 15 MHz
    bbConfigureIQ(handle, 2, 15.0e6);
    // By default the sentinel is zero, set to -1
    bbConfigureIQTriggerSentinel(TRIGGER_SENTINEL);

    // Initiate the device, once this function returns the device
    // will be streaming I/Q.
    status = bbInitiate(handle, BB_STREAMING, BB_STREAM_IQ);
    if(status != bbNoError) {
        std::cout << "Initiate error\n";
        std::cout << bbGetErrorString(status) << "\n";
        bbCloseDevice(handle);
        return;
    }

    // Get I/Q stream characteristics
    double sampleRate, bandwidth;
    bbQueryIQParameters(handle, &sampleRate, &bandwidth);

    // This is how much data we want after the external trigger
    const int N = 1e6;

    // I/Q capture buffer
    std::vector<std::complex<float>> buffer(N);
    // We only care about the first trigger we see. If you need more triggers,
    //   this can be an array.
    int triggerPos = 0;

    while(true) {
        // Perform the capture, pass NULL for any parameters we don't care about
        status = bbGetIQUnpacked(handle, &buffer[0], N, &triggerPos, 1,
            BB_FALSE, 0, 0, 0, 0);

        // At this point, N I/Q samples have been retrieved and
        //  stored in the buffer array. If any triggers were seen during the capture
        //  of the returned samples, the trigger parameter will contain an index into
        //  the buffer array at which the trigger was seen. 
        // A trigger value not equal to the sentinel means a trigger is present
        if(triggerPos != TRIGGER_SENTINEL) {
            // We have a trigger, now finish capture
            break;
        }
    }

    // Unless trigger was at beginning, we still need to capture some data
    //  to retrive N samples.
    int samplesAfterTrigger = N - triggerPos;
    int samplesLeft = N - samplesAfterTrigger;
    
    // Move the samples after the trigger to the beginning of the buffer.
    for(int i = 0; i < samplesAfterTrigger; i++) {
        buffer[i] = buffer[i + triggerPos];
    }

    // Get the rest of the contiguous samples
    bbGetIQUnpacked(handle, &buffer[samplesAfterTrigger], samplesLeft,
        0, 0, BB_FALSE, 0, 0, 0, 0);

    // When done, stop streaming and close device.
    bbAbort(handle);
    bbCloseDevice(handle);
}

/** [extTriggerExample1] */
