#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sp_api.h"

// Open a device
// Configure and perform a series of sweeps using the I/Q sweep list capabilities.
// Configure an I/Q sweep list and perform many sweeps using the queued sweep
//   mechanism for maximum performance and throughput.
// For a simpler example, see the 'single' sweep example.

void sp_example_iq_sweep_list_queue()
{
    int handle = -1;

    // Open device
    SpStatus status = spOpenDevice(&handle);

    if(status != spNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // Sweep from 400MHz to 3GHz in 40MHz steps
    // The bandwidth of the USB SP145 is 40MHz
    // 66 steps covers the full range (inclusive)
    const int steps = 66;

    // The data returned should be corrected, scaled to sqrt(mW) instead of full scale.
    spSetIQSweepListCorrected(handle, spTrue);
    // Returne the data at 32-bit floating point complex values
    spSetIQSweepListDataType(handle, spDataType32fc);
    // Set frequency steps
    spSetIQSweepListSteps(handle, steps);

    // If the GPS antenna is connected, this will instruct the device to
    // discipline to the internal GPS PPS. This will improve frequency 
    // and timestamp accuracy.
    spSetGPSTimebaseUpdate(handle, spTrue);

    // Number of samples to capture at each frequency
    const int samplesPerStep = 16384;

    // Steps have center freq 400M, 440M, 480M, ..., 3000M
    for(int i = 0; i < steps; i++) {
        spSetIQSweepListFreq(handle, i, 400.0e6 + 40.0e6 * i);
        spSetIQSweepListRef(handle, i, -20.0);
        spSetIQSweepListSampleCount(handle, i, samplesPerStep);
    }

    // Total samples between all steps
    const int totalSamples = steps * samplesPerStep;

    // Configure the device
    spConfigure(handle, spModeIQSweepList);

    // Allocate memory for the capture and timestamps
    // Need an array for the data and timestamps for every sweep we plan on queuing
    std::vector<std::complex<float>> iq[16];
    std::vector<int64_t> timestamps[16];
    for(int i = 0; i < 16; i++) {
        iq[i].resize(totalSamples);
        timestamps[i].resize(steps);
    }

    // The total number of sweeps to perform and process
    const int sweepsToPerform = 1000;

    // Start by queuing up all 16 sweeps. Can have at most 16 sweeps active.
    for(int i = 0; i < 16; i++) {
        spIQSweepListStartSweep(handle, i, &iq[i][0], &timestamps[i][0]);
    }

    // Now finish 1000 sweeps, starting up a new sweep for each sweep we finish,
    //   which keeps at least 16 sweeps in the queue
    for(int i = 0; i < sweepsToPerform; i++) {
        int pos = i % 16;

        status = spIQSweepListFinishSweep(handle, pos);
        if(status != spNoError) {
            printf("%s\n", spGetErrorString(status));
        }

        // The sweep at pos is now finished
        // The data at iq[pos] and timestamps[i] is ready
        // It is recommended to either process the data now before starting another
        //   acquisition. 

        // GPS lock doesn't occur immediately upon opening. If
        //   the GPS is cold it could take several minutes to acquire lock. If warm, it 
        //   might not lock for several seconds. Generally the hardware will need
        //   to see at least 1 PPS after opening before lock can be determined. 
        //   It will take multiple PPS after opening for disciplining to be achieved.
        // Call the spGetGPSState function to determine if the timestamps were returned
        //   under GPS lock.

        // Put this sweep back into the queue
        spIQSweepListStartSweep(handle, pos, &iq[pos][0], &timestamps[pos][0]);
    }

    // We are done, finish all outstanding sweeps
    for(int i = 0; i < 16; i++) {
        spIQSweepListFinishSweep(handle, i);
    }

    // Done with device
    spCloseDevice(handle);
}
