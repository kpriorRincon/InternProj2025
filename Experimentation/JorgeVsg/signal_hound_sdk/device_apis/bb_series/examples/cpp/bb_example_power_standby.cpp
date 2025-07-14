#include "bb_api.h"
#include <iostream>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example shows how to set the BB60D to the low power standby state.
The BB60C/A do not have a low power standby mode.
*/

void bbExamplePowerStandby()
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

    // Do some measurement here
    // ...

    // Stop measurement before continuing.
    bbAbort(handle);

    status = bbSetPowerState(handle, bbPowerStateStandby);
    if(status != bbNoError) {
        std::cout << "Failed to set power state to standby\n";
        std::cout << bbGetErrorString(status) << std::endl;
    }

    // Devices is now in standby mode, do something else for awhile

    status = bbSetPowerState(handle, bbPowerStateOn);
    if(status != bbNoError) {
        std::cout << "Failed to set power state to on\n";
        std::cout << bbGetErrorString(status) << std::endl;
    }

    // Continue with measurements
    // ...

    bbCloseDevice(handle);
}