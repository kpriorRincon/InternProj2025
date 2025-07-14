/** [realTimeExample1] */

// Configure the device for real-time spectrum analysis, and retrieve the real-time sweeps and frames.

#include <cstdio>
#include <cstdlib>

#include "sm_api.h"

static void checkStatus(SmStatus status)
{
    if(status > 0) { // Warning
        printf("Warning: %s\n", smGetErrorString(status));
        return;
    } else if(status < 0) { // Error
        printf("Error: %s\n", smGetErrorString(status));
        exit(-1);
    }
}

void sm_example_real_time()
{
    int handle = -1;
    SmStatus status = smNoError;

    // Uncomment this to open a USB SM device
    status = smOpenDevice(&handle);
    // Uncomment this to open a networked SM device with a default network config
    //status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    // Check open status
    checkStatus(status);

    // Configure the measurement
    smSetRefLevel(handle, -20.0); // -20dBm reference level
    smSetRealTimeCenterSpan(handle, 2.45e9, 160.0e6); // 160MHz span at 2.45GHz center freq
    smSetRealTimeRBW(handle, 30.0e3); // 30kHz min RBW with Nuttall window
    smSetRealTimeDetector(handle, smDetectorMinMax);
    smSetRealTimeScale(handle, smScaleLog, -20.0, 100.0); // On the frame, ref of -20, 100dB height
    smSetRealTimeWindow(handle, smWindowNutall);

    // Initialize the measurement
    status = smConfigure(handle, smModeRealTime);
    checkStatus(status);

    // Get the configured measurement parameters as reported by the receiver
    double actualRBW, actualStart, binSize, poi;
    int sweepSize, frameWidth, frameHeight;

    smGetRealTimeParameters(handle, &actualRBW, &sweepSize, &actualStart,
        &binSize, &frameWidth, &frameHeight, &poi);

    // Create memory for our sweep and frame
    float *sweep = new float[sweepSize];
    float *frame = new float[frameWidth * frameHeight];

    // Retrieve a series of sweeps/frames
    for(int i = 0; i < 100; i++) {
        // Retrieve just the color frame and max sweep.
        smGetRealTimeFrame(handle, frame, nullptr, nullptr, sweep, nullptr, nullptr);

        // Do something with data here
    }

    // Done with the device
    smAbort(handle);
    smCloseDevice(handle);

    // Clean up
    delete [] sweep;
    delete [] frame;
}

/** [realTimeExample1] */