#include "bb_api.h"
#include <iostream>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example illustrates opening a device and retrieving basic device
info and diagnostics.
*/

void bbExampleGetDeviceInfo()
{
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << "Issue opening device\n";
        std::cout << bbGetErrorString(status) << "\n";
        return;
    }

    uint32_t serialNumber;
    bbGetSerialNumber(handle, &serialNumber);

    int deviceType;
    bbGetDeviceType(handle, &deviceType);

    int firmwareVersion;
    bbGetFirmwareVersion(handle, &firmwareVersion);

    float temp, voltage, current;
    bbGetDeviceDiagnostics(handle, &temp, &voltage, &current);

    std::cout << "Using API version " << bbGetAPIVersion() << "\n";
    std::cout << "Serial number " << serialNumber << "\n";
    if(deviceType == BB_DEVICE_BB60D) {
        std::cout << "Device type BB60D\n";
    } else if(deviceType == BB_DEVICE_BB60C) {
        std::cout << "Device type BB60C\n";
    } else if(deviceType == BB_DEVICE_BB60A) {
        std::cout << "Device type BB60A\n";
    }
    std::cout << "Firmware version " << firmwareVersion << "\n";
    std::cout << "Device temp " << temp << " C\n";
    std::cout << "Device voltage " << voltage << " V\n";
    std::cout << "Device current " << current << " mA\n";

    bbCloseDevice(handle);
}