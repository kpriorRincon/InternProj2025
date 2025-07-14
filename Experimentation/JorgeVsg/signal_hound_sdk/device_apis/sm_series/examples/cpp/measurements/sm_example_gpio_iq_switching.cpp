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

void sm_example_gpio_iq_switching()
{
    int handle = -1;
    SmStatus status = smNoError;

    // Uncomment this to open a USB SM device
    status = smOpenDevice(&handle);
    // Uncomment this to open a networked SM device with a default network config
    //status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    checkStatus(status);

    // Configure the receiver for IQ acquisition
    smSetRefLevel(handle, -20.0);
    smSetIQCenterFreq(handle, 900.0e6); // 900MHz
    smSetIQSampleRate(handle, 1); // 50 / 1 = 25MS/s IQ 
    smSetIQBandwidth(handle, smTrue, 20.0e6); // 20MHz of bandwidth

    // Now lets configure the auto GPIO switching states
    // Switch between 4 antenna every 2.5ms
    // Cycle through 4 antennas at a rate of 100Hz
    const int states = 4;
    uint8_t gpioStates[states];
    uint32_t counts[states];
    for(int i = 0; i < states; i++) {
        gpioStates[i] = i;
        counts[i] = 125000;
    }
    status = smSetGPIOSwitching(handle, gpioStates, counts, states);
    checkStatus(status);

    // Initialize the receiver with the above settings
    status = smConfigure(handle, smModeIQ);
    checkStatus(status);

    // At this point, the device is active streaming IQ data and the GPIO switching
    // has been enabled. You can call smGetIQ with buffers for both IQ data and
    // external trigger positions. External triggers will be inserted every time the
    // GPIO state has returned to the first position. In this case you will see 100
    // triggers per second. The triggers allow you to line up the IQ data with the
    // GPIO configurations.
    // See the other C++ examples for calling the smGetIQ function and finding the ext.
    // triggers.

    // Finished
    smAbort(handle);
    smCloseDevice(handle);
}