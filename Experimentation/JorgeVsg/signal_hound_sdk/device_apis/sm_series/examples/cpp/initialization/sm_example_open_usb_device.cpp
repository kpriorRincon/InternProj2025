#include <cstdio>
#include <cstdlib>

#include "sm_api.h"

// This function only applies to SM200A/B devices.
// Retrieve a list of all USB SM200 devices connected to the PC.
// Opens the first device found in the list.
void sm_example_open_usb_device()
{
    int serialList[SM_MAX_DEVICES] = {0};
    int deviceCount = 0;

    smGetDeviceList(serialList, &deviceCount);

    if(deviceCount <= 0) {
        printf("No devices found\n");
        return;
    }

    printf("Found %d devices\n", deviceCount);
    for(int i = 0; i < deviceCount; i++) {
        printf("Device %d Serial Number: %d\n", i+1, serialList[i]);
    }

    int targetDeviceSerial = serialList[0];

    // Open the first SM device found
    int handle = -1;
    SmStatus status = smOpenDeviceBySerial(&handle, targetDeviceSerial);

    if(status != smNoError) {
        printf("Unable to open device\n");
        exit(-1);
    } 

    printf("Device opened successfully\n");

    // Just for fun verify the serial we opened is one we targetted
    int openedSerial = 0;
    smGetDeviceInfo(handle, 0, &openedSerial);
    if(openedSerial != targetDeviceSerial) {
       printf("Error opening target serial number\n");
    }

    // Finished
    smCloseDevice(handle);
}