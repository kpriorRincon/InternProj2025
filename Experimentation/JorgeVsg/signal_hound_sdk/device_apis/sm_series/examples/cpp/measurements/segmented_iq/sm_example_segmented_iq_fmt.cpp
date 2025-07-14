#include "sm_api.h"

#include <vector>

// SM200B/SM435B only.
// This example demonstrates setting up a single frequency
//  mask triggered I/Q capture using the segmented I/Q captures of the
//  SM200B and SM435B.

void sm_example_segmented_iq_fmt()
{
    // Number of I/Q samples
    const int PRETRIGGER_LEN = 32768;
    const int POSTTRIGGER_LEN = 262144;
    const int CAPTURE_LEN = PRETRIGGER_LEN + POSTTRIGGER_LEN;

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
    status = smSetSegIQDataType(handle, smDataType32fc);
    status = smSetSegIQCenterFreq(handle, 2.45e9);

    // Configure the frequency mask
    // Create a triangle shaped frequency mask, that is centered around zero Hz.
    // Ramp up threshold (center - 100MHz) from -80 to 0dBm.
    // Ramp down (center + 100MHz) from 0 to -80dBm.
    std::vector<double> fmtFrequencies, fmtAmplitudes;
    fmtFrequencies.push_back(-100.0e6);
    fmtFrequencies.push_back(   0.0e6);
    fmtFrequencies.push_back(+100.0e6);
    fmtAmplitudes.push_back(-80.0);
    fmtAmplitudes.push_back(  0.0);
    fmtAmplitudes.push_back(-80.0);
    // Tell the API to use a 8192 point FFT mask.
    // The larger the FMT mask size, the more resolution in the frequency domain
    //  but the less resolution in the time domain.
    status = smSetSegIQFMTParams(handle, 8192, &fmtFrequencies[0], &fmtAmplitudes[0], fmtFrequencies.size());

    // Setup a single segment capture 
    status = smSetSegIQSegmentCount(handle, 1);
    status = smSetSegIQSegment(handle, 0, smTriggerTypeFMT, PRETRIGGER_LEN, POSTTRIGGER_LEN, 2.0);

    status = smConfigure(handle, smModeIQSegmentedCapture);
    if(status != smNoError) {
        // Unable to configure device
        const char *errStr = smGetErrorString(status);
        return;
    }

    // 2 floats for each I/Q sample
	std::vector<float> buf(CAPTURE_LEN * 2);
    int64_t nsSinceEpoch = 0;
    SmBool timedOut;

    // Start the measurement, the analyzer will begin looking for an external
    //   trigger at this point.
    status = smSegIQCaptureStart(handle, 0);

    // Notify some other device to transmit/trigger if necessary.

    // Block until complete
    status = smSegIQCaptureWait(handle, 0);
    // Did the capture timeout?
    status = smSegIQCaptureTimeout(handle, 0, 0, &timedOut);
    // Read the timestamp
    status = smSegIQCaptureTime(handle, 0, 0, &nsSinceEpoch);
    // Read the I/Q data
    status = smSegIQCaptureRead(handle, 0, 0, (float*)buf.data(), 0, buf.size() / 2);
    status = smSegIQCaptureFinish(handle, 0);

    smAbort(handle);
    smCloseDevice(handle);
}
