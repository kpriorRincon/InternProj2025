#include <cstdio>
#include <cstdlib>

#include "sm_api.h"

// GPIO sweep functionality enables you to set the GPIO at various points
// during the sweep. You can specify the frequencies at which the GPIO
// changes during a sweep. 

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

void sm_example_gpio_intra_sweep()
{
    int handle = -1;
    SmStatus status = smNoError;

    // Uncomment this to open a USB SM device
    status = smOpenDevice(&handle);
    // Uncomment this to open a networked SM device with a default network config
    //status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    checkStatus(status);

    // Set all 8 GPIO bits to output
    status = smSetGPIOState(handle, smGPIOStateOutput, smGPIOStateOutput);
    checkStatus(status);

    // Here we set up the GPIO frequency breaks
    // We set up at which frequencies we want to switch to different
    // GPIO outputs. In this example we imagine an antenna assembly for 4 antennas
    // that have different frequency ranges. 
    const int gpioStepCount = 4;
    SmGPIOStep gpioSteps[gpioStepCount];
    gpioSteps[0].freq = 100.0e3; // 100k to 900M
    gpioSteps[0].mask = 0x0;
    gpioSteps[1].freq = 900.0e6; // 900M to 3G
    gpioSteps[1].mask = 0x1;
    gpioSteps[2].freq = 3.0e9; // 3G to 10G
    gpioSteps[2].mask = 0x2;
    gpioSteps[3].freq = 10.0e9; // 10G to 20G
    gpioSteps[3].mask = 0x3;

    // Tell the API about the GPIO sweep settings
    smSetGPIOSweep(handle, gpioSteps, gpioStepCount);

    // Now we configure the device for a sweep
    // Configure the receiver sweep functionality
    smSetRefLevel(handle, 0.0); // 0 dBm reference level
    smSetSweepCenterSpan(handle, 100.0e3, 20.0e9); // Full span
    smSetSweepCoupling(handle, 300.0e3, 300.0e3, 0.001); // 300kHz rbw/vbw, 1ms acquisition
    smSetSweepDetector(handle, smDetectorMinMax, smVideoPower); // average power detector
    smSetSweepScale(handle, smScaleLog); // return sweep in dBm
    smSetSweepWindow(handle, smWindowFlatTop);
    smSetSweepSpurReject(handle, smFalse); // No software spur reject
    smSetSweepSpeed(handle, smSweepSpeedNormal);

    // Initialize the device for sweep measurement mode
    status = smConfigure(handle, smModeSweeping);
    checkStatus(status);

    // Get the configured sweep parameters as reported by the receiver
    double actualRBW, actualVBW, actualStartFreq, binSize;
    int sweepSize;
    smGetSweepParameters(handle, &actualRBW, &actualVBW, &actualStartFreq, &binSize, &sweepSize);

    // Create memory for our sweep
    float *sweep = new float[sweepSize];

    // Get the sweep
    // GPIO switching will be active for the sweep
    smGetSweep(handle, nullptr, sweep, nullptr);
    smAbort(handle);

    // Do something with data

    // Finished
    smCloseDevice(handle);
}