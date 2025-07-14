#include "sm_api.h"

#include <vector>

// SM200B/SM435B only.
// This example demonstrates setting up a single immediate triggered I/Q capture
//  using the 160MHz I/Q capture capabilities of the SM200B/SM435B. This examples uses the
//  full start/stop capture routines. This is often needed for more complex measurement
//  scenarios or ones which required buffering several captures or ones with more than
//  a single triggered segment.

void sm_example_segmented_iq_imm_manual()
{
    // Number of I/Q samples, 50 million, 1/5th of a second
    const int CAPTURE_LEN = 50e6;

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
    // Setup a single segment capture
    status = smSetSegIQSegmentCount(handle, 1);
    status = smSetSegIQSegment(handle, 0, smTriggerTypeImm, 0, CAPTURE_LEN, 0.0);

    status = smConfigure(handle, smModeIQSegmentedCapture);
    if(status != smNoError) {
        // Unable to configure device
        const char *errStr = smGetErrorString(status);
        return;
    }

    // 2 floats for each I/Q sample
	std::vector<float> buf(CAPTURE_LEN * 2);
    int64_t nsSinceEpoch = 0;

    // Start the measurement
    status = smSegIQCaptureStart(handle, 0);
    // Block until complete, can also use the async wait function
    status = smSegIQCaptureWait(handle, 0);
    // Read the time, if a GPS is connected, the time returned will be GPS time,
    //   otherwise, system time is reported.
    status = smSegIQCaptureTime(handle, 0, 0, &nsSinceEpoch);
    // Get the I/Q data
    status = smSegIQCaptureRead(handle, 0, 0, (float*)buf.data(), 0, buf.size() / 2);
    // Finish the measurement. This function cleans up any resources associated with
    //   the capture and makes it available to start again.
    status = smSegIQCaptureFinish(handle, 0);

    // Do something with data

    smAbort(handle);
    smCloseDevice(handle);
}
