// This examples demonstrates how to achieve the sustained 1THz sweep throughput.
// It does this by ensuring up to N traces are actively queued. This function
// sweeps 5THz of bandwidth and measures the time taken to do so.
// Sweep speed is reported at the end.
//
// Additionally shows how to calculate the 100% POI for the configured measurement.
// Each sweep is tagged in the same location with a system clock (250MHz) and so the
//   time between each acquired sweep can be measured at a 4ns resolution.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <list>

#if defined(_WIN32)
#include <Windows.h>
inline uint64_t GetSystemTicks() {
    return GetTickCount64();
}
#else
inline uint64_t GetSystemTicks() {
    timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return (uint64_t)tv.tv_sec * 1000 + (uint64_t)tv.tv_nsec / 1e6;
}
#endif

#include "sm_api.h"

void sm_example_thz_sweep()
{
    int handle = -1;
    SmStatus status = smNoError;

    // Uncomment this to open a USB SM device
    status = smOpenDevice(&handle);
    // Uncomment this to open a networked SM device with a default network config
    status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    if(status != smNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // Configure the receiver sweep functionality
    // Things to note for THz sweep speeds
    // smSweepSpeedFast must be set
    // RBW must equal VBW and be above 30kHz for Nuttall window
    smSetSweepSpeed(handle, smSweepSpeedFast);
    smSetRefLevel(handle, -20.0); // -20dBm reference level
    smSetSweepCenterSpan(handle, 10.5e9, 19.0e9); // 1G to 20GHz range
    smSetSweepStartStop(handle, 1.0e9, 20.0e9);
    smSetSweepCoupling(handle, 100.0e3, 100.0e3, 0.001); // 100kHz rbw/vbw
    smSetSweepDetector(handle, smDetectorMinMax, smVideoPower); // min/max power detector
    smSetSweepScale(handle, smScaleLog); // return sweep in dBm
    smSetSweepWindow(handle, smWindowNutall);
    smSetSweepSpurReject(handle, smFalse); // No software spur reject in fast sweep
    smSetPreselector(handle, smFalse); // No preselector in fast sweep

    // Initialize the device for sweep measurement mode
    status = smConfigure(handle, smModeSweeping);
    if(status != smNoError) {
        printf("Unable to configure device\n");
        printf("%s\n", smGetErrorString(status));
        smCloseDevice(handle);
        exit(-1);
    }

    // Get the configured sweep parameters as reported by the receiver
    double actualRBW, actualVBW, actualStartFreq, binSize;
    int sweepSize;
    smGetSweepParameters(handle, &actualRBW, &actualVBW, &actualStartFreq, &binSize, &sweepSize);

    // Create memory for our sweep
    float *sweep = new float[sweepSize];

    // To maintain 1THz sweep speed, we can keep several sweeps queued
    // A good rule of thumb is to keep at least 2-4 sweeps queued or ~40GHz worth of
    //   spectrum queued, whichever is more. Only 16 total sweeps can be queued.
    double spectrumToSweep = 5 * 1.0e12; // N * THz worth of spectrum, or !N seconds
    int sweepsToQueue = 4;
    int queuePos = 0;
    double sweepSpan = binSize * sweepSize; // Calculate span of sweep
    int sweepsToPerform = spectrumToSweep / sweepSpan; // How many sweeps for 5THz of spectrum

    // Variables for calculating 100% POI
    int64_t lastSweepTime = 0; // Used to meausure delta sweep times
    std::list<int64_t> sweepTimes;

    uint64_t start = GetSystemTicks();

    printf("Starting THZ sweep speed test\n");

    // Start all sweeps in queue
    for(int i = 0; i < sweepsToQueue; i++) {
        smStartSweep(handle, i);
    }

    // Acquire sweeps
    // Loop through queue collecting sweeps, ensuring to queue them back
    //  up when we are done.
    while(sweepsToPerform > 0) {
        // Finish the sweep at this position
        int64_t currSweepTime;
        status = smFinishSweep(handle, queuePos, nullptr, sweep, &currSweepTime);
        if(status != smNoError) {
            printf("Finish sweep status: %s\n", smGetErrorString(status));
        }

        // If you don't care about the POI calculation skip this
        // Add to our list of sweep times
        if(lastSweepTime != 0) {
            int64_t deltaSweepTime = currSweepTime - lastSweepTime;
            sweepTimes.push_back(deltaSweepTime);
        }
        lastSweepTime = currSweepTime;

        // Process/store sweep here
        // Sweep in contained in the 'sweep' array

        // Start it back up
        smStartSweep(handle, queuePos);

        // Increase and wrap our queue pos
        queuePos++;
        if(queuePos >= sweepsToQueue) {
            queuePos = 0;
        }

        sweepsToPerform--;
    }

    // We still have sweeps in the queue, finish them out
    // Technically it doesn't matter which order we finish them out
    for(int i = 0; i < sweepsToQueue; i++) {
        smFinishSweep(handle, i, nullptr, sweep, nullptr);
    }

    uint64_t elapsed = GetSystemTicks() - start;

    double sweepSpeed = spectrumToSweep / ((double)elapsed/1000.0); // In Hz
    printf("Sweep speed = %f GHz/s\n", sweepSpeed / 1.0e9);

    // Calculate our 100% POI by taking the worst sweep time over our sweep times.
    // By using the worst sweep time we can gaurantee any event longer than our worst
    //  sweep time must have been intercepted.
    int64_t maxSweepTime = *std::max_element(sweepTimes.begin(), sweepTimes.end());
    printf("100%% POI = %.3f ms\n", (double)maxSweepTime / 1.0e6);

    // Done with the device
    smCloseDevice(handle);

    // Clean up
    delete [] sweep;
}
