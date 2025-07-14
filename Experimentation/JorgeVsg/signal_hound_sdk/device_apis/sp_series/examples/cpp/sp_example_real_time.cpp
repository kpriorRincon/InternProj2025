/** [realTimeExample1] */

#include <cstdio>
#include <cstdlib>

#include "sp_api.h"

// Configure the device for real-time spectrum analysis, and retrieve the real-time sweeps and frames.

static void checkStatus(SpStatus status)
{
    if(status > 0) { // Warning
        printf("Warning: %s\n", spGetErrorString(status));
        return;
    } else if(status < 0) { // Error
        printf("Error: %s\n", spGetErrorString(status));
        exit(-1);
    }
}

void sp_example_real_time()
{
    int handle = -1;
    SpStatus status = spNoError;

    status = spOpenDevice(&handle);

    // Check open status
    checkStatus(status);

    // Configure the measurement
    spSetRefLevel(handle, -20.0); // -20dBm reference level
    spSetRealTimeCenterSpan(handle, 2.45e9, 40.0e6); // 40MHz span at 2.45GHz center freq
    spSetRealTimeRBW(handle, 30.0e3); // 30kHz min RBW with Nuttall window
    spSetRealTimeDetector(handle, spDetectorMinMax);
    spSetRealTimeScale(handle, spScaleLog, -20.0, 100.0); // On the frame, ref of -20, 100dB height
    spSetRealTimeWindow(handle, spWindowNutall);

    // Initialize the measurement
    status = spConfigure(handle, spModeRealTime);
    checkStatus(status);

    // Get the configured measurement parameters as reported by the receiver
    double actualRBW, actualStart, binSize, poi;
    int sweepSize, frameWidth, frameHeight;

    spGetRealTimeParameters(handle, &actualRBW, &sweepSize, &actualStart,
        &binSize, &frameWidth, &frameHeight, &poi);

    // Create memory for our sweep and frame
    float *sweep = new float[sweepSize];
    float *frame = new float[frameWidth * frameHeight];

    // Retrieve a series of sweeps/frames
    for(int i = 0; i < 100; i++) {
        // Retrieve just the color frame and max sweep.
        spGetRealTimeFrame(handle, frame, nullptr, nullptr, sweep, nullptr, nullptr);

        // Do something with data here
    }

    // Done with the device
    spAbort(handle);
    spCloseDevice(handle);

    // Clean up
    delete [] sweep;
    delete [] frame;
}

/** [realTimeExample1] */

