#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sm_api.h"

// This example illustrates how to use the smGetIQFullBandSweep.
// This is a special measurement mode available on the SM200 for capturing very short
//   acquisitions at the full baseband rate.
// For basic sweeps or for I/Q streaming, use different examples.

void sm_example_full_band_iq_sweep()
{
    int handle = -1;

    // Open a USB SM device
    SmStatus status = smOpenDevice(&handle);
    // Open a networked SM device
    //SmStatus status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    if(status != smNoError) {
        printf("Unable to open device\n");
        printf("%s\n", smGetErrorString(status));
        exit(-1);
    }

    // Configuration parameters for a full sweep from 600M to 20GHz
    // atten of 0 = full sensitivity
    int atten = 0;
    // I/Q Samples per frequency
    int samplesPerStep = 32768;
    // First freq at (16+1)*39.0625MHz = 664.0625MHz
    int startIndex = 16;
    // Step the center frequency 9 * 39.0625MHz = 351.5625MHz
    int stepSize = 9;
    // Number of center frequencies to capture data at
    int steps = 56;

    // First configure the measurement
    smSetIQFullBandAtten(handle, atten); 
    // Enable I/Q imbalance and flatness corrections
    smSetIQFullBandCorrected(handle, smTrue);
    smSetIQFullBandSamples(handle, samplesPerStep);
    // Only immediate trigger is availabe for I/Q full band sweeps
    // Setting any other value here will be ignored when performing the sweep
    smSetIQFullBandTriggerType(handle, smTriggerTypeImm);

    // Allocate memory for full capture. 
    // Memory is stored as I/Q, so 2 floating point values per sample
    std::vector<float> iq(2 * samplesPerStep * steps);

    status = smGetIQFullBandSweep(handle, (float*)&iq[0], startIndex, stepSize, steps); 
    if(status != smNoError) {
        printf("%s\n", smGetErrorString(status));
    }

    // If the sweep returned successfully, I/Q samples are stored contiguously in the I/Q array
    // Accessing the first sample at a given step N (zero-based) would look like
    // iq[2 * samplesPerStep * N]
    // The device is idle after returning and the sweep is ready to be configured (if desired)
    //   and performed again.

    smCloseDevice(handle);
}