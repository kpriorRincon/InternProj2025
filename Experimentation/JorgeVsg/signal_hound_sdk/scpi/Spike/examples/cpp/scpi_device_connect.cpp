#include <cassert>
#include <cstdio>
#include <vector>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates using VISA to connect to Spike and if a device
//   is not currently active in the Spike software, connecting it.
// For this example to be fully illustrative, it is necessary to disconnect the
//   device manually in the Spike software with File->Disconnect Device prior to running
//   this example.

void scpi_device_connect()
{
    ViSession rm, inst;
    ViStatus rmStatus;

    // Get the VISA resource manager
    rmStatus = viOpenDefaultRM(&rm);
    assert(rmStatus == 0);

    // Open a session to the Spike software, Spike must be running at this point
    ViStatus instStatus = viOpen(rm, "TCPIP::localhost::5025::SOCKET", VI_NULL, VI_NULL, &inst);
    assert(instStatus == 0);

    // For SOCKET programming, we want to tell VISA to use a terminating character 
    //   to end a read operation. In this case we want the newline character to end a 
    //   read operation. The termchar is typically set to newline by default. Here we
    //   set it for illustrative purposes.
    viSetAttribute(inst, VI_ATTR_TERMCHAR_EN, VI_TRUE);
    viSetAttribute(inst, VI_ATTR_TERMCHAR, '\n');

    // Determine whether there is already an active device
    int activeDevice;
    viQueryf(inst, "SYST:DEV:ACT?\n", "%d", &activeDevice);

    if(activeDevice == 1) {
        // Do nothing if already active
        printf("Device already connected\n");
    } else {
        // No device active, lets connect the first one we find

        // Ask how many devices Spike sees connected to the PC.
        int deviceCount;
        viQueryf(inst, "SYST:DEV:COUN?\n", "%d", &deviceCount);

        if(deviceCount < 1) {
            printf("No devices found\n");
        } else {
            // Get the list of all available serial numbers
            // LIST? returns a comma separated list of serial numbers. If there
            //   is only one device present on the PC, it will be a single serial number.
            // Preallocate vector for all the serial numbers, and scan them in 1 by 1.
            std::vector<int> deviceList(deviceCount);
            viQueryf(inst, "SYST:DEV:LIST?\n", "%,#d", &deviceCount, &deviceList[0]);
            
            // Setting a 25 second timeout. Just for the device connect command.
            // The longest connect time of any Signal Hound device is 20 seconds. This will
            //   account for the worst case. You can also call the viScanf function in a 
            //   loop while it returns timeout. This will give you a finer resolution on 
            //   breakout conditions.
            viSetAttribute(inst, VI_ATTR_TMO_VALUE, 25e3);

            // Now lets open the first device
            int openSuccess;
            viQueryf(inst, "SYST:DEVICE:CONNECT? %d\n", "%d", deviceList[0], &openSuccess);
            if(!openSuccess) {
                // Device failed to open
                printf("Device failed to open\n");
            }

            // Set timeout back to 2 seconds
            viSetAttribute(inst, VI_ATTR_TMO_VALUE, 2e3);
        }
    }

    // Query the active device
    char idn[256];
    viPrintf(inst, "*IDN?\n");
    viScanf(inst, "%s", idn);

    // Print the IDN response
    printf("*IDN? Response: %s\n", idn);

    // Done
    viClose(inst);
}