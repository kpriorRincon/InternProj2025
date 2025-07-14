/** [iqSweepListExample1] */

// Configure and perform a single sweep in the I/Q sweep list mode.
// Configure a sweep with 3 different frequencies and different capture sizes at each frequency. 
// Shows how to index the sweep.
// For a more basic example, see the 'simple' example.
// For an example which queues multiple sweeps, see the 'queue' example.

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sm_api.h"

void sm_example_iq_sweep_list_single()
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
    // 3 frequency steps
    smSetIQSweepListSteps(handle, 3);

    // If the GPS antenna is connected, this will instruct the device to
    // discipline to the internal GPS PPS. This will improve frequency 
    // and timestamp accuracy.
    smSetGPSTimebaseUpdate(handle, smTrue);

    // Configure all three frequency steps

    // 1GHz, 1000 samples to be collected
    smSetIQSweepListFreq(handle, 0, 1.0e9);
    smSetIQSweepListRef(handle, 0, -20.0);
    smSetIQSweepListSampleCount(handle, 0, 1000);

    // 2GHz, 3000 samples to be collected
    smSetIQSweepListFreq(handle, 1, 2.0e9);
    smSetIQSweepListRef(handle, 1, -20.0);
    smSetIQSweepListSampleCount(handle, 1, 2000);

    // 3GHz, 3000 samples to be collected
    smSetIQSweepListFreq(handle, 2, 3.0e9);
    smSetIQSweepListRef(handle, 2, -20.0);
    smSetIQSweepListSampleCount(handle, 2, 3000);

    // Total samples between all 3 frequency steps
    const int totalSamples = 6000;

    // Configure the device
    smConfigure(handle, smModeIQSweepList);

    // Allocate memory for the capture
    std::vector<std::complex<float>> iq(totalSamples);

    // Memory for the timestamps
    int64_t timestamps[3];

    // Perform the sweep
    smIQSweepListGetSweep(handle, &iq[0], timestamps);

    // Example of how to index the data
    // Get pointers to the data for the 3 steps
    std::complex<float> *step1 = &iq[0];
    std::complex<float> *step2 = &iq[1000];
    std::complex<float> *step3 = &iq[3000];

    // Do something with the data here

    // The three timestamps will be the times of the samples at 
    //   step1[0], step2[0], and step3[0]

    // GPS lock doesn't occur immediately upon opening. If
    //   the GPS is cold it could take several minutes to acquire lock. If warm, it 
    //   might not lock for several seconds. Generally the hardware will need
    //   to see at least 1 PPS after opening before lock can be determined. 
    //   It will take multiple PPS after opening for disciplining to be achieved.
    // Call the smGetGPSState function to determine if the timestamps were returned
    //   under GPS lock.

    // Done with device
    smCloseDevice(handle);
}

/** [iqSweepListExample1] */
