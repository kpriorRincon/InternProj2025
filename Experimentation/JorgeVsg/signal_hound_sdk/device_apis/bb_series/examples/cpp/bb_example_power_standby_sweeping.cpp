#include "bb_api.h"
#include <iostream>
#include <vector>

#include <Windows.h>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example shows how to set the BB60D to the low power standby state in
between sweeps. This prevents the need to re-initiate the device for sweeps
each time you want to set the device into a low power state. This cannot
be done in streaming measurements such as real-time, I/Q streaming, or audio
demodulation.

Changing the power state is not instantaneous and has diminishing returns
below 100ms in standby mode.

The BB60C/A does not have a lower power standby mode.
*/

void bbExamplePowerStandbySweeping()
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

    if(deviceType != BB_DEVICE_BB60D) {
        std::cout << "This device does not support the standby power state.\n";
        bbCloseDevice(handle);
        return;
    }

    // Lets configure our sweeps
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

    // Lets do 10 sweeps with a low duty cycle and set the device to a low power
    //   state between each sweep.
    for(int i = 0; i < 10; i++) {
        status = bbFetchTrace_32f(handle, sweepSize, 0, sweep.data());
        if(status != bbNoError) {
            std::cout << "Sweep status: " << bbGetErrorString(status) << "\n";
        }

        // Go to standby 
        status = bbSetPowerState(handle, bbPowerStateStandby);
        if(status != bbNoError) {
            std::cout << "Failed to set power state to standby\n";
            std::cout << bbGetErrorString(status) << std::endl;
        }

        // Devices is now in standby mode
        // Do something with sweep, or something else for awhile
        Sleep(1000);

        // Set power state back to on for next sweep
        status = bbSetPowerState(handle, bbPowerStateOn);
        if(status != bbNoError) {
            std::cout << "Failed to set power state to on\n";
            std::cout << bbGetErrorString(status) << std::endl;
        }
    }

    bbCloseDevice(handle);
}