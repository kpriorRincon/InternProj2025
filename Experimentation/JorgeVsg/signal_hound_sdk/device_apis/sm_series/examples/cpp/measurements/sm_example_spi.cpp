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

// Convenience union
union SpiWord {
    uint32_t word;
    uint8_t bytes[4];
};
static_assert(sizeof(SpiWord) == 4, "Invalid size");

void sm_example_spi()
{
    int handle = -1;
    SmStatus status = smNoError;

    // Uncomment this to open a USB SM device
    status = smOpenDevice(&handle);
    // Uncomment this to open a networked SM device with a default network config
    //status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    checkStatus(status);

    uint32_t bytes = 0x76543210;
    
    // Transmit 0x10 
    status = smWriteSPI(handle, bytes, 1);
    // Transmit 0x3210 
    status = smWriteSPI(handle, bytes, 2);
    // Transmit 0x76543210
    status = smWriteSPI(handle, bytes, 4);
    
    // Using the convenience union
    SpiWord spi;
    spi.bytes[0] = 0x10;
    spi.bytes[1] = 0x32;
    spi.bytes[2] = 0x54;
    spi.bytes[3] = 0x76;

    // Transmit 0x10 
    status = smWriteSPI(handle, spi.word, 1);
    // Transmit 0x76543210
    status = smWriteSPI(handle, spi.word, 4);

    // Finished
    smCloseDevice(handle);
}