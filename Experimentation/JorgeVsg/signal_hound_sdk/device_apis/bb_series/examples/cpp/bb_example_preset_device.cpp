#include "bb_api.h"
#include <iostream>

#include <Windows.h>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example illustrates how to use the bbPreset function to power cycle a device.
This example assumes a single BB60 is connected to the PC.
*/

void bbExamplePresetDevice()
{
    // Setup, open device and retreive serial number
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << "Issue opening device\n";
        std::cout << bbGetErrorString(status) << "\n";
        return;
    }

    uint32_t serialNumber;
    bbGetSerialNumber(handle, &serialNumber);

    // Here we preset the device
    // This function sends the reset command to the device.
    // If the device is no longer receiving USB communications, it will
    //   not recieve the command.
    bbPreset(handle);
    // Calling this function is necessary as it frees all resources associated
    //   with the BB60 and releases the handle.
    bbCloseDevice(handle);

    // Now we wait 3 seconds for the device to reset and reenumerate
    // On some systems this may need to be longer.
    Sleep(3 * 1000);

    // Now we can try to reopen the device using the serial we captured.
    status = bbOpenDeviceBySerialNumber(&handle, serialNumber);
    
    // If status is not bbNoError here, you might consider waiting longer
    //   and retrying.

    // You can interface the device here is openDevice passed

    bbCloseDevice(handle);
}