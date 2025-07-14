/** [sweepExample1] */

#include "bb_api.h"
#include <iostream>
#include <vector>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example demonstrates using the API to perform a sweep
*/

void bbExampleSweep()
{
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << "Issue opening device\n";
        std::cout << bbGetErrorString(status) << "\n";
        return;
    }

    // Configure a sweep from 850MHz to 950MHz with a 10kHz
    //  RBW/VBW and an expected input of at most -20dBm.
    bbConfigureRefLevel(handle, -20.0);
    bbConfigureCenterSpan(handle, 900.0e6, 100.0e6);
    bbConfigureSweepCoupling(handle, 10.0e3, 10.0e3, 0.001, BB_RBW_SHAPE_FLATTOP, BB_NO_SPUR_REJECT);
    bbConfigureAcquisition(handle, BB_AVERAGE, BB_LOG_SCALE);
    bbConfigureProcUnits(handle, BB_POWER);

    // Configuration complete, initialize the device
    status = bbInitiate(handle, BB_SWEEPING, 0);
    if(status < bbNoError) {
        std::cout << "Error configuring device\n";
        std::cout << bbGetErrorString(status);
        exit(-1);
    } 

    // Get sweep characteristics and allocate memory for sweep
    uint32_t sweepSize;
    double binSize, startFreq;
    bbQueryTraceInfo(handle, &sweepSize, &binSize, &startFreq);

    std::vector<float> sweep(sweepSize);

    // Get the sweep
    // Pass NULL for the min parameter. For most scenarios, the min sweep can be ignored.
    status = bbFetchTrace_32f(handle, sweepSize, 0, sweep.data());
    if(status != bbNoError) {
        std::cout << "Sweep status: " << bbGetErrorString(status) << "\n";
    }

    // If the sweep status is not an error, the sweep is now stored in the sweep
    //   vector. The frequency of the first index in the vector is startFreq.
    // To get the frequency of any bin in the vector use the equation
    // freqHz = startFreq + binSize * index;

    // From this point, you can continue to get sweeps with this configuration
    //   by calling bbFetchTrace again, or reconfigure and initiate the
    //   device for a new sweep configuration.

    // Finished/close device
    bbAbort(handle);
    bbCloseDevice(handle);
}

/** [sweepExample1] */
