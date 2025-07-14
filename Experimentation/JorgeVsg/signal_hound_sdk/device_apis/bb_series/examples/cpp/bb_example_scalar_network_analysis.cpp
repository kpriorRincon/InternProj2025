/** [snaExample1] */

#include "bb_api.h"
#include <iostream>
#include <vector>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

// This example demonstrates how to use the API to perform a single tracking generator sweep.
// See the manual for a full description of each step of the process in the
// Scalar Network Analysis section.

void bbExampleScalarNetworkAnalysis()
{
    // Connect device
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << "Issue opening device\n";
        std::cout << bbGetErrorString(status) << "\n";
        return;
    }

    if(bbAttachTg(handle) != bbNoError) {
        std::cout << "Unable to find tracking generator\n";
        return;
    }

    // Sweep some device at 900 MHz center with 1 MHz span
    bbConfigureCenterSpan(handle, 900.0e6, 1.0e6);
	bbConfigureAcquisition(handle, BB_MIN_AND_MAX, BB_LOG_SCALE);
    bbConfigureRefLevel(handle, -10.0);
    bbConfigureSweepCoupling(handle, 1.0e3, 1.0e3, 0.001, BB_RBW_SHAPE_FLATTOP, BB_SPUR_REJECT);
	bbConfigureProcUnits(handle, BB_POWER);

    // Additional configuration routine
    // Configure a 100 point sweep
    // The size of the sweep is a suggestion to the API, it will attempt to get near the requested size
    // Optimized for high dynamic range and passive devices
    bbConfigTgSweep(handle, 100, true, true);

	// Configuration complete, initialize the device
    status = bbInitiate(handle, BB_TG_SWEEPING, 0);
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

    // Create test set-up without DUT present
    // Get one sweep
	// Pass NULL for the min parameter. For most scenarios, the min sweep can be ignored.
    status = bbFetchTrace_32f(handle, sweepSize, 0, sweep.data());
    if(status != bbNoError) {
        std::cout << "Sweep status: " << bbGetErrorString(status) << "\n";
		return;
    }

    // Store baseline
    bbStoreTgThru(handle, TG_THRU_0DB);

    // Should pause here, and insert DUT into test set-up
    bbFetchTrace_32f(handle, sweepSize, 0, sweep.data());

    // From here, you can sweep several times without needing to restore the thru.
    // Once you change your setup, you should reconfigure the device and 
    //   store the thru again without the DUT inline.

	// Finished/close device
    bbAbort(handle);
    bbCloseDevice(handle);
}

/** [snaExample1] */
