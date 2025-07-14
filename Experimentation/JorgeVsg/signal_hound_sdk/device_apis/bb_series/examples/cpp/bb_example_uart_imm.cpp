#include "bb_api.h"
#include <iostream>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example shows how to configure the BB60D to output 1 byte of data 
on the UART output port. The device must be idle (not actively making measurements)
to do this. 
*/

void bbExampleUARTImm()
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
        std::cout << "Only BB60D devices support UART output\n";
        bbCloseDevice(handle);
        exit(-1);
    }

    // Configure the IO for UART output
    bbConfigureIO(handle, BB60D_PORT1_DISABLED, BB60D_PORT2_OUT_UART);

    // Set 1MHz UART baud rate
    bbSetUARTRate(handle, BB60D_UART_BAUD_1000K);

    // Transmit a byte
    bbWriteUARTImm(handle, 0xAA);

    // Done
    bbCloseDevice(handle);
}