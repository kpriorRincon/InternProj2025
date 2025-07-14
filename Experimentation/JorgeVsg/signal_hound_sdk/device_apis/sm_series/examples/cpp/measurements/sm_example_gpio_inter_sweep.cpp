#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sm_api.h"

// Inter-sweep GPIO functionality enabled you to set the GPIO in between
//   sweeps. Commonly used with queued sweeps, you can rapidly cycle between
//   different GPIO settings for your desired sweep range.
// This example illustrates sweeping a range 16 times with 16 different GPIO
//   settings.

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

void sm_example_gpio_inter_sweep()
{
    int device, sweep;
    SmStatus status;

    // Open device, get handle
    status = smOpenDevice(&device);
    assert(status == smNoError);

    // Set all 8 GPIO pins to output (this is default)
    smSetGPIOState(device, smGPIOStateOutput, smGPIOStateOutput);

    // Configure device for fast sweep mode
    smSetRefLevel(device, -20.0);
    smSetPreselector(device, smFalse);
    smSetSweepSpeed(device, smSweepSpeedFast);
    smSetSweepCenterSpan(device, 10.0e9, 5.0e9);
    smSetSweepCoupling(device, 300.0e3, 300.0e3, 0.001);
    smSetSweepDetector(device, smDetectorMinMax, smVideoPower);
    smSetSweepScale(device, smScaleLog);
    smSetSweepWindow(device, smWindowNutall);
    smSetSweepSpurReject(device, smFalse);

    smConfigure(device, smModeSweeping);

    // Query sweep parameters
    double actualRBW, actualVBW, actualStartFreq, binSize;
    int sweepSize;
    smGetSweepParameters(device, &actualRBW, &actualVBW, &actualStartFreq, &binSize, &sweepSize);

    // Number of sweeps to perform
    const int queuesz = 16;

    // Allocated memory for sweeps
    std::vector<float> sweepArrays[queuesz];
    for(int i = 0; i < queuesz; i++) {
        sweepArrays[i].resize(sweepSize);
    }

    // Use GPIO outputs [0-15]
    uint8_t gpioSettings[queuesz];
    for(int i = 0; i < queuesz; i++) {
        gpioSettings[i] = i;
    }

    // Start the sweeps, setting the GPIO prior to starting each sweep.
    // The actual GPIO switch doesn't occur until the sweep actually starts.
    for(int i = 0; i < queuesz; i++) {
        smSetSweepGPIO(device, i, gpioSettings[i]);
        smStartSweep(device, i);
    }

    // Finish all the sweeps
    for(int i = 0; i < queuesz; i++) {
        smFinishSweep(device, i, nullptr, &sweepArrays[i][0], nullptr);
    }

    // Perform user processing here

    smAbort(device);
    smCloseDevice(device);
}
