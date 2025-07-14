#include <cstdio>
#include <cstdlib>

#include "sm_api.h"

static void checkStatus(SmStatus status)
{
    if(status > 0) { // Warning
        printf("Warning: %s\n", smGetErrorString(status));
        return;
    } else if(status < 0) { // Error
        printf("Error: %s\n", smGetErrorString(status));
        exit(-1);
    }
}

void sm_example_gpio_imm()
{
    int handle = -1;
    SmStatus status = smNoError;

    // Uncomment this to open a USB SM device
    status = smOpenDevice(&handle);
    // Uncomment this to open a networked SM device with a default network config
    //status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    checkStatus(status);

    // Set the in/out state of the GPIO pins
    // Set the lower 4 pins to inputs, and the upper 4 pins to outputs
    status = smSetGPIOState(handle, smGPIOStateInput, smGPIOStateOutput);
    checkStatus(status);

    // Lets set the upper 4 pins to bits 1010
    // Note: because the lower 4 pins are set as inputs, the lower 4 bits
    // provided to this function are ignored. We send zero bits in this example. 
    status = smWriteGPIOImm(handle, 0xA0);
    checkStatus(status);

    // Lets read the lower 4 pins.
    // Note: The lower 4 pins will be read since they are set to inputs. 
    // The bits returned for the upper 4 pins will be returned as they are set.
    // Which means the upper 4 pins will read as 1010 since we set them above.
    uint8_t pins = 0;
    status = smReadGPIOImm(handle, &pins);
    checkStatus(status);

    // You can also set the pins in a loop with a delay
    // In this example loop, we are cycling through the 16 possible states for the
    // upper 4 output pins.
    for(int i = 0; i < 16; i++) {
        // Sleep here

        smWriteGPIOImm(handle, i << 4);
    }

    // In this example loop we are sweeping and reading the GPIO pins after
    //  each sweep. (The GPIO input pins are read after each sweep)
    for(int i = 0; i < 100; i++) {
        // Sweep here
        smReadGPIOImm(handle, &pins);
    }

    // Finished
    smCloseDevice(handle);
}