#include <cstdio>
#include <cstdlib>

#include "sp_api.h"

// Retrieve a list of all SP145 devices connected to the PC,
// then open the first device found in the list.
void sp_example_open_device()
{
    int serialList[SP_MAX_DEVICES] = {0};
    int deviceCount = 8;

    spGetDeviceList(serialList, &deviceCount);

    if(deviceCount <= 0) {
        printf("No devices found\n");
        return;
    }

    printf("Found %d devices\n", deviceCount);
    for(int i = 0; i < deviceCount; i++) {
        printf("Device %d Serial Number: %d\n", i+1, serialList[i]);
    }

    int targetDeviceSerial = serialList[0];

    // Open the first SP device found
    int handle = -1;
    SpStatus status = spOpenDeviceBySerial(&handle, targetDeviceSerial);

    if(status != spNoError) {
        printf("Unable to open device\n");
        exit(-1);
    } 

    printf("Device opened successfully\n");

    // Verify the serial we opened is one we targetted
    int openedSerial = 0;
    spGetSerialNumber(handle, &openedSerial);
    if(openedSerial != targetDeviceSerial) {
       printf("Error opening target serial number\n");
    }

    // Finished
    spCloseDevice(handle);
}