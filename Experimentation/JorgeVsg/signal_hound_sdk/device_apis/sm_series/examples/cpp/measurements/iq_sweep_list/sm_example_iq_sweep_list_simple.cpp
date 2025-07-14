#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sm_api.h"

// Open a device
// Configure and perform a single sweep in the I/Q sweep list mode.
// Configure a sweep which acquires data at one frequency.
// Does not utilize timestamps.
// For a sweep which acquires at several frequencies and uses timestamps,
//   see the 'single' example.

void sm_example_iq_sweep_list_simple()
{
    int handle = -1;

    // Open a USB SM device
    SmStatus status = smOpenDevice(&handle);
    // Open a networked SM device
    //SmStatus status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    if(status != smNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // The data returned should be corrected, scaled to sqrt(mW) instead of full scale.
    smSetIQSweepListCorrected(handle, smTrue);
    // Returne the data at 32-bit floating point complex values
    smSetIQSweepListDataType(handle, smDataType32fc);
    // Acquire I/Q data at a single frequency
    smSetIQSweepListSteps(handle, 1);

    // Configure the acquisition
    const int totalSamples = 1000;

    // 1GHz, 1000 samples to be collected
    smSetIQSweepListFreq(handle, 0, 1.0e9);
    smSetIQSweepListRef(handle, 0, -20.0);
    smSetIQSweepListSampleCount(handle, 0, totalSamples);

    // Configure the device
    smConfigure(handle, smModeIQSweepList);

    // Allocate memory for the capture
    std::vector<std::complex<float>> iq(totalSamples);

    // Perform the sweep, pass null for the timestamps
    smIQSweepListGetSweep(handle, &iq[0], nullptr);

    // The I/Q array will now contain totalSamples number of
    //   I/Q samples centered at 1GHz

    // Done with device
    smCloseDevice(handle);
}
