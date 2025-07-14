#include <cstdio>
#include <cstdlib>

#include "sp_api.h"

// Open the first SP145 found on the system and print basic
// diagnostic information about the unit.

void sp_example_get_device_info()
{
    int handle = -1;
    SpStatus status = spNoError;

    // Open the first SP145 device found
    status = spOpenDevice(&handle);

    if(status != spNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // Query and print information
    printf("API Version: %s\n", spGetAPIVersion());

    int serialNumber = 0;
    spGetSerialNumber(handle, &serialNumber);
    printf("Serial Number: %d\n", serialNumber);

    int major, minor, rev;
    spGetFirmwareVersion(handle, &major, &minor, &rev);
    printf("Firmware Version: %d.%d.%d\n", major, minor, rev);

    float voltage, current, temperature;
    spGetDeviceDiagnostics(handle, &voltage, &current, &temperature);
    printf("Device Temperature: %f Celsius\n", temperature);
    printf("Device Power Consumption: %f W\n", current * voltage);

    // Finished
    spCloseDevice(handle);
}
