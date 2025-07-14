/** [sweepExample1] */

// Configure the device for sweeps and perform a single sweep.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sm_api.h"

void sm_example_sweep()
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

    // Configure the sweep
    smSetRefLevel(handle, -20.0); // -20dBm reference level
    smSetSweepCenterSpan(handle, 2.45e9, 100.0e6); // ISM band
    smSetSweepCoupling(handle, 10.0e3, 10.0e3, 0.001); // 10kHz rbw/vbw, 1ms acquisition
    smSetSweepDetector(handle, smDetectorAverage, smVideoPower); // average power detector
    smSetSweepScale(handle, smScaleLog); // return sweep in dBm
    smSetSweepWindow(handle, smWindowFlatTop);
    smSetSweepSpurReject(handle, smFalse); // No software spur reject

    // Initialize the device for sweep measurement mode
    status = smConfigure(handle, smModeSweeping);
    if(status != smNoError) {
        printf("Unable to configure device\n");
        printf("%s\n", smGetErrorString(status));
        smCloseDevice(handle);
        exit(-1);
    }

    // Get the configured sweep parameters as reported by the receiver
    double actualRBW, actualVBW, actualStartFreq, binSize;
    int sweepSize;
    smGetSweepParameters(handle, &actualRBW, &actualVBW, &actualStartFreq, &binSize, &sweepSize);

    // Create memory for our sweep
    std::vector<float> sweep(sweepSize);

    // Get sweep, ignore the min sweep and the sweep time
    status = smGetSweep(handle, nullptr, sweep.data(), nullptr);
    if(status != smNoError) {
        printf("Sweep status: %s\n", smGetErrorString(status));
    }

    // Done with the device
    smCloseDevice(handle);
}

/** [sweepExample1] */