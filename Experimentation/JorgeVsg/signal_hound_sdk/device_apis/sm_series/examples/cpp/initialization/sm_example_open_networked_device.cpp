#include <cstdio>
#include <cstdlib>

#include "sm_api.h"

// This function only applies to SM200C (networked) devices.
// This code assumes the device is connected to a valid 10GbE network
//   interface that has been configured properly (see our NIC setup guides). 
// The SM200C uses a default IP and port, this code also assumes the device
//   IP and port has not been modified.
void sm_example_open_networked_device()
{
    int handle = -1;
    SmStatus status = smOpenNetworkedDevice(&handle,
        SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    if(status != smNoError) {
        printf("Issue opening device\n");
        printf("%s\n", smGetErrorString(status));
        exit(-1);
    } 

    // At this point the device is open, lets retrieve some
    // information about the device.
    int serial = 0;
    smGetDeviceInfo(handle, 0, &serial);
    printf("Device serial number: %d\n", serial);

    int major, minor, rev;
    smGetFirmwareVersion(handle, &major, &minor, &rev);
    printf("Device firmware version: %d.%d.%d\n", major, minor, rev);

    // Finished
    smCloseDevice(handle);
}