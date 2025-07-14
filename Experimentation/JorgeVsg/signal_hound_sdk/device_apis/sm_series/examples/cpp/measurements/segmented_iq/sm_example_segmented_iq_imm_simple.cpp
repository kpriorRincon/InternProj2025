/** [iqSegmentedExample1] */

// SM200B/SM435B only.
// This example demonstrates setting up a single immediate triggered I/Q capture
//  using the 160MHz I/Q capture capabilities of the SM200B/SM435B. This examples uses the
//  convenience function for completing the capture. See the "imm_manual" example for a full
//  example.

#include "sm_api.h"

#include <vector>

void sm_example_segmented_iq_imm_simple()
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
    SmBool timedOut = smFalse;

    // This example uses the convenience function for completing the capture.
    // See the manual example for performing the full sequence of capture functions.
    // Immediate triggered acquisitions can't time out, so we ignore the timeout.
    status = smSegIQCaptureFull(handle, 0, buf.data(), 0, CAPTURE_LEN, &nsSinceEpoch, &timedOut);

    // Do something with data

    smAbort(handle);
    smCloseDevice(handle);
}

/** [iqSegmentedExample1] */