#include "bb_api.h"
#include <iostream>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example illustrates how to configure the receiver to accept an external
10MHz reference. Which device type is connected will affect how you configure
the receiver.
*/ 

void bbExampleRefIn()
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

    if(deviceType == BB_DEVICE_BB60D) {
        // BB60D devices
        // Port 1 is AC coupled 
        bbConfigureIO(handle, BB60D_PORT1_10MHZ_REF_IN, BB60D_PORT2_DISABLED);
    } else {
        // BB60C/A devices
        // Configure an AC coupled 10MHz ref in on port 1
        bbConfigureIO(handle, BB60C_PORT1_10MHZ_REF_IN | BB60C_PORT1_AC_COUPLED, 0);
    }

    // All measurements taken will now be performed while the receiver is 
    //   disciplined with the external 10MHz.

    bbCloseDevice(handle);
}