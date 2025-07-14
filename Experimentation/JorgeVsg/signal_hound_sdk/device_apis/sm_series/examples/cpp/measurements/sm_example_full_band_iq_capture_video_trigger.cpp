#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sm_api.h"

// This example illustrates how to use the smGetIQFullBand() function.
// This is a special measurement mode available on the SM for capturing very short
//   acquisitions at the full baseband rate.
// For basic I/Q streaming, see other examples.
// This example requires an SM200C with FW version 7.7.5 or greater. 
// If not using a device that meets these requirements, video triggering and
//   triggering timeout will not be available.

void sm_example_full_band_iq_capture_video_trigger()
{
    int handle = -1;
    SmStatus status = smNoError;

    // Open an SM200C device
    status = smOpenNetworkedDevice(&handle, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);

    if(status != smNoError) {
        printf("Unable to open device\n");
        printf("%s\n", smGetErrorString(status));
        exit(-1);
    }

    const int samples = 32768;

    // First configure the measurement
    // 0dB attenuation, full sensitivity
    smSetIQFullBandAtten(handle, 0); 
    // Enable I/Q imbalance and flatness corrections
    smSetIQFullBandCorrected(handle, smTrue);
    // Retrieve the full 32k samples available
    smSetIQFullBandSamples(handle, samples);
    // Enable video trigger, with 100ms timeout and -20dBFS level
    smSetIQFullBandTriggerType(handle, smTriggerTypeVideo);
    smSetIQFullBandVideoTrigger(handle, -20.0);
    smSetIQFullBandTriggerTimeout(handle, 0.1);

    // Allocate memory for full capture. 
    // Data is retrieved as I/Q, so 2 floating point values per sample is required.
    std::vector<float> iq(2 * samples);

    // Specify a frequency index and center frequency for the capture.
    int freqIndex = 26;
    double centerFreq = (26 + 1) * 39.0625e6;

    // Perform the measurement
    status = smGetIQFullBand(handle, &iq[0], 16);
    if(status != smNoError) {
        printf("%s\n", smGetErrorString(status));
    }

    // If the capture returned successfully, I/Q samples are stored contiguously in
    //  the I/Q vector. The device is idle after returning and the full band capture
    //  is ready to be configured (if needed) and performed again.

    smCloseDevice(handle);
}