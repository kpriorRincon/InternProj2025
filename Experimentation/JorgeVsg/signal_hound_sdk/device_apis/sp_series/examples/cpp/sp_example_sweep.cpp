/** [sweepExample1] */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sp_api.h"

// Configure the device for sweeps and perform a single sweep.
void sp_example_sweep()
{
    int handle = -1;
    SpStatus status = spNoError;

    status = spOpenDevice(&handle);

    // Check open status
    if (status != spNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // Configure the sweep
    spSetRefLevel(handle, -20.0); // -20dBm reference level
    spSetSweepCenterSpan(handle, 2.45e9, 100.0e6); // ISM band
    spSetSweepCoupling(handle, 10.0e3, 10.0e3, 0.001); // 10kHz rbw/vbw, 1ms acquisition
    spSetSweepDetector(handle, spDetectorAverage, spVideoPower); // average power detector
    spSetSweepScale(handle, spScaleLog); // return sweep in dBm
    spSetSweepWindow(handle, spWindowFlatTop);

    // Initialize the device for sweep measurement mode
    status = spConfigure(handle, spModeSweeping);
    if (status != spNoError) {
        printf("Unable to configure device\n");
        printf("%s\n", spGetErrorString(status));
        spCloseDevice(handle);
        exit(-1);
    }

    // Get the configured sweep parameters as reported by the receiver
    double actualRBW, actualVBW, actualStartFreq, binSize;
    int sweepSize;
    spGetSweepParameters(handle, &actualRBW, &actualVBW, &actualStartFreq, &binSize, &sweepSize);

    // Create memory for our sweep
    std::vector<float> sweep(sweepSize);

    // Get sweep, ignore the min sweep and the sweep time
    status = spGetSweep(handle, nullptr, sweep.data(), nullptr);
    if (status != spNoError) {
        printf("Sweep status: %s\n", spGetErrorString(status));
    }

    // Done with the device
    spCloseDevice(handle);
}

/** [sweepExample1] */