#include "sm_api.h"

#include <complex>
#include <vector>

// SM200B/SM435B only.
// This example demonstrates using the LTE resample function for segmented I/Q captures.
// The LTE resample function resamples data sequentially retrieved from a segmented I/Q
//   capture from the native 250MS/s rate to a LTE friendly 245.76MS/s
// This example configures a large single immediate triggered I/Q capture. Since the 
//   capture is large, the capture must be read sequentially in smaller sizes
//   and then resampled to the new rate. Captures that can be read fully in one read
//   are much simpler to resample since they only require calling the LTE resample 
//   function once.

typedef std::complex<float> cplx32f;

void sm_example_segmented_iq_lte_resample()
{
    // Number of I/Q samples
    const int CAPTURE_LEN = 10.0e6;

	int handle = -1;
    SmStatus status = smOpenDevice(&handle);

    if(status != smNoError) {
        // Unable to open device
        const char *errStr = smGetErrorString(status);
        return;
    }

    // Verify device has segmented I/Q capture capability
    SmDeviceType deviceType;
    smGetDeviceInfo(handle, &deviceType, 0);
    if(deviceType != smDeviceTypeSM200B && deviceType != smDeviceTypeSM435B) {
        // Invalid device type
        smCloseDevice(handle);
        return;
    }

    // Set device reference level, maximum expected input signal
    status = smSetRefLevel(handle, 0.0);

    // Configure the 160MHz capture
    // 16-bit signed shorts as capture type
    status = smSetSegIQDataType(handle, smDataType32fc);
    status = smSetSegIQCenterFreq(handle, 1.0e9);
    // Setup a single segment capture with immediate trigger
    status = smSetSegIQSegmentCount(handle, 1);
    status = smSetSegIQSegment(handle, 0, smTriggerTypeImm, 0, CAPTURE_LEN, 0.0);

    status = smConfigure(handle, smModeIQSegmentedCapture);
    if(status != smNoError) {
        // Unable to configure device
        const char *errStr = smGetErrorString(status);
        return;
    }

    // Read the capture this many samples at a time
    const int bufSize = 1.0e6;
	std::vector<cplx32f> buf(bufSize);
    // Will contain the final resampled data, since we are downsampling by a small
    //  fraction, we will always end up with less data than we started with, safe 
    //  to allocate a slightly larger amount.
    std::vector<cplx32f> resampledBuf(CAPTURE_LEN);

    // Start the measurement
    status = smSegIQCaptureStart(handle, 0);
    // Block until complete
    status = smSegIQCaptureWait(handle, 0);

    // Read through the buffer and read all the I/Q, and store it to disk
    int samplesRead = 0;
    int outputPtr = 0;
    bool firstRead = true;

    // Loop through the full buffer reading at most bufSize worth of I/Q
    //   samples on each iteration.
    while(samplesRead < CAPTURE_LEN) {
        int samplesLeftToRead = (CAPTURE_LEN - samplesRead);
        int samplesToRead = (samplesLeftToRead > bufSize) ? bufSize : samplesLeftToRead;

        status = smSegIQCaptureRead(handle, 0, 0, &buf[0], samplesRead, samplesToRead);

        int outputLen = resampledBuf.size() - outputPtr;
        status = smSegIQLTEResample((float*)&buf[0], buf.size(), (float*)&resampledBuf[outputPtr], &outputLen, firstRead);

        samplesRead += samplesToRead;
        outputPtr += outputLen;
        firstRead = false;
    }

    status = smSegIQCaptureFinish(handle, 0);

    // At this point the resampledBuf contains 'outputPtr' complex resampled I/Q values
    //   at the 245.76MS/s rate.

    smAbort(handle);
    smCloseDevice(handle);
}
