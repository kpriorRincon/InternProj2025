#include <cstdio>
#include <cstdlib>

#include "sm_api.h"

// Open the first SM200A found on the system and print basic
// diagnostic information about the unit.

void sm_example_get_device_info()
{
    int handle = -1;
    SmStatus status = smNoError;

    // Open the first USB SM200 device found
    status = smOpenDevice(&handle);
    // Open a networked SM200 device with default network config
    //status = smOpenNetworkedDevice(&handle, SM200_ADDR_ANY, SM200_DEFAULT_ADDR, SM200_DEFAULT_PORT);

    if(status != smNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // Query and print information
    printf("API Version: %s\n", smGetAPIVersion());

    int serialNumber = 0;
    smGetDeviceInfo(handle, 0, &serialNumber);
    printf("Serial Number: %d\n", serialNumber);

    int major, minor, rev;
    smGetFirmwareVersion(handle, &major, &minor, &rev);
    printf("Firmware Version: %d.%d.%d\n", major, minor, rev);

    float voltage, current, temperature;
    smGetDeviceDiagnostics(handle, &voltage, &current, &temperature);
    printf("Device Temperature: %f Celsius\n", temperature);
    printf("Device Power Consumption: %f W\n", current * voltage);

    // Finished
    smCloseDevice(handle);
}
