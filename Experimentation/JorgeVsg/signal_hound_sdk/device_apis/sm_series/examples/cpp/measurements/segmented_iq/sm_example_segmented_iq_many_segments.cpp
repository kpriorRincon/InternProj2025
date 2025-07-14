#include "sm_api.h"

#include <vector>

// SM200B/SM435B only.
// This example demonstrates configuring a several segment capture using the 160MHz
//   I/Q capture capabilities of the SM200B/SM435B.
// A capture can be composed of several triggered captures, called segments.
// Each segment can have a different trigger type.
// The benefit of configured one capture of several segments vs many captures of one
//   segment, is that all acquisition is guaranteed to occur back to back. When configuring
//   several captures, USB latencies might mean the device isn't fully setup for the next
//   capture.
// The concept of several segments per capture and many captures being buffered can be combined
//   if needed.

void sm_example_segmented_iq_many_segments()
{
    // For simplicity, all segments in this capture will use the same pre/post 
    //   trigger sizes.
    const int PRETRIGGER_LEN = 32768;
    const int POSTTRIGGER_LEN = 32768;
    const int CAPTURE_LEN = PRETRIGGER_LEN + POSTTRIGGER_LEN;

    const int SEGMENT_COUNT = 4;

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
    status = smSetSegIQCenterFreq(handle, 1.0e9);

    // Setup an rising edge video trigger at -40 dBm
    status = smSetSegIQVideoTrigger(handle, -40.0, smTriggerEdgeRising);
    // Setup the external trigger
    status = smSetSegIQExtTrigger(handle, smTriggerEdgeRising);

    // Setup a 4 segment capture
    // In our hypothetical scenario we are looking for an external trigger, followed
    //  by 3 video triggers.
    // The result is 4 triggered segments in a single capture.
    status = smSetSegIQSegmentCount(handle, SEGMENT_COUNT);
    status = smSetSegIQSegment(handle, 0, smTriggerTypeExt, PRETRIGGER_LEN, POSTTRIGGER_LEN, 2.0);
    status = smSetSegIQSegment(handle, 1, smTriggerTypeVideo, PRETRIGGER_LEN, POSTTRIGGER_LEN, 2.0);
    status = smSetSegIQSegment(handle, 2, smTriggerTypeVideo, PRETRIGGER_LEN, POSTTRIGGER_LEN, 2.0);
    status = smSetSegIQSegment(handle, 3, smTriggerTypeVideo, PRETRIGGER_LEN, POSTTRIGGER_LEN, 2.0);

    status = smConfigure(handle, smModeIQSegmentedCapture);
    if(status != smNoError) {
        // Unable to configure device
        const char *errStr = smGetErrorString(status);
        return;
    }

    // Allocate memory for the capture
	std::vector<float> buf(CAPTURE_LEN * 2);

    // Perform a single capture of 4 segments
    smSegIQCaptureStart(handle, 0);

    // Block until complete
    status = smSegIQCaptureWait(handle, 0);

    // At this point the entire capture (all segments) are ready.
    // Loop through them

    for(int i = 0; i < SEGMENT_COUNT; i++) {
        // Get the segment timeout status
        SmBool timedOut = smFalse;
        status = smSegIQCaptureTimeout(handle, 0, i, &timedOut);
        // Get the segment time.
        int64_t nsSinceEpoch = 0;
        status = smSegIQCaptureTime(handle, 0, i, &nsSinceEpoch);
        // Read the I/Q data for that segment
        status = smSegIQCaptureRead(handle, 0, i, (float*)buf.data(), 0, buf.size() / 2);
    }

    // Complete the capture
    status = smSegIQCaptureFinish(handle, 0);

    smAbort(handle);
    smCloseDevice(handle);
}
