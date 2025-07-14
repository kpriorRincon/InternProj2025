/** [sweepExample2] */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sp_api.h"

// Configure the device for sweeps and perform a single sweep.
void sp_example_sweep_ant_switch()
{
    int handle = -1;
    SpStatus status = spNoError;

    status = spOpenDevice(&handle);

    // Check open status
    if(status != spNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // Configure the GPIO port for antenna switching 
    spSetGPIOPort(handle, SpGPIOFunctionUARTSweep);
    spSetUARTBaudRate(handle, 460.8e3);

    // Configure the sweep
    spSetRefLevel(handle, -20.0); // -20dBm reference level
    spSetSweepStartStop(handle, 100.0e3, 14.5e9); // Full span
    spSetSweepCoupling(handle, 100.0e3, 100.0e3, 0.001); // 10kHz rbw/vbw
    spSetSweepDetector(handle, spDetectorAverage, spVideoPower); // average power detector
    spSetSweepScale(handle, spScaleLog); // return sweep in dBm
    spSetSweepWindow(handle, spWindowFlatTop);

    // Configure the frequency switch points
    const int N = 3;
    double antFreqs[N] = { 0.0, 200.0e6, 6.0e9 };
    uint8_t antData[N] = { 0, 1, 2 };
    spSetSweepGPIOSwitching(handle, antFreqs, antData, N);

    // Initialize the device for sweep measurement mode
    status = spConfigure(handle, spModeSweeping);
    if(status != spNoError) {
        printf("Unable to configure device\n");
        printf("%s\n", spGetErrorString(status));
        spCloseDevice(handle);
        exit(-1);
    }

    // Get the configured sweep parameters as reported by the receiver
    double actualRBW, actualVBW, actualStartFreq, binSize;
    int sweepSize;
    spGetSweepParameters(handle, &actualRBW, &actualVBW, &actualStartFreq, &binSize, &sweepSize);

    // Create memory for our sweep
    std::vector<float> sweep(sweepSize);

    // Get sweep in loop, monitor GPIO port for UART writes
    while(true) {
        status = spGetSweep(handle, nullptr, sweep.data(), nullptr);
        if(status != spNoError) {
            printf("Sweep status: %s\n", spGetErrorString(status));
        }
        if(status < spNoError) {
            break;
        }
    }

    // Done with the device
    spCloseDevice(handle);
}

/** [sweepExample2] */