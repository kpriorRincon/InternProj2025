#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sp_api.h"

// Open a device
// Configure and perform a single sweep in the I/Q sweep list mode.
// Configure a sweep which acquires data at one frequency.
// Does not utilize timestamps.
// For a sweep which acquires at several frequencies and uses timestamps,
//   see the 'single' example.

void sp_example_iq_sweep_list_simple()
{
    int handle = -1;

    // Open device
    SpStatus status = spOpenDevice(&handle);

    if(status != spNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // The data returned should be corrected, scaled to sqrt(mW) instead of full scale.
    spSetIQSweepListCorrected(handle, spTrue);
    // Returne the data at 32-bit floating point complex values
    spSetIQSweepListDataType(handle, spDataType32fc);
    // Acquire I/Q data at a single frequency
    spSetIQSweepListSteps(handle, 1);

    // Configure the acquisition
    const int totalSamples = 1000;

    // 1GHz, 1000 samples to be collected
    spSetIQSweepListFreq(handle, 0, 1.0e9);
    spSetIQSweepListRef(handle, 0, -20.0);
    spSetIQSweepListSampleCount(handle, 0, totalSamples);

    // Configure the device
    spConfigure(handle, spModeIQSweepList);

    // Allocate memory for the capture
    std::vector<std::complex<float>> iq(totalSamples);

    // Perform the sweep, pass null for the timestamps
    spIQSweepListGetSweep(handle, &iq[0], nullptr);

    // The I/Q array will now contain totalSamples number of
    //   I/Q samples centered at 1GHz

    // Done with device
    spCloseDevice(handle);
}
