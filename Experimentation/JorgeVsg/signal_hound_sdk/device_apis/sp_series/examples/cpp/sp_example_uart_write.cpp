#include "sp_api.h"

#include <cstdio>

// Configure the device GPIO port for UART writes.
// Write bytes in loop.
void sp_example_uart_write()
{
    int handle = -1;
    SpStatus status = spOpenDevice(&handle);
    if(status != spNoError) {
        printf("Unable to open device\n");
        return;
    }

    // Configure the GPIO port for direct UART writes
    spSetGPIOPort(handle, SpGPIOFunctionUARTDirect);

    // Configure the UART baud rate and print the actual rate used by the device
    spSetUARTBaudRate(handle, 460.8e3);
    float actualRate;
    spGetUARTBaudRate(handle, &actualRate);
    printf("Actual Baud Rate: %f\n", actualRate);

    // Loop forever, writing incrementing values
    uint8_t i = 0;
    while(true) {
        spWriteUARTDirect(handle, i++);
    }

    spCloseDevice(handle);
}
