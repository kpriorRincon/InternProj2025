#include "sm_api.h"

#include <vector>

// SM200B/SM435B only.
// This example demonstrates buffering many short video triggered acquisitions.
// Each capture is a single video triggered segment.
// This allows you to buffer and capture many short trigger events in a sparse RF
//   environment. This is useful for bursty communications signals, radar, or
//   interference hunting.
// This example buffered the maximum amount of video triggered sweeps. If no input
//   pulses are found, each capture will timeout sequentially.

void sm_example_segmented_iq_buffered_captures()
{
    // I/Q samples in capture
    const int PRETRIGGER_LEN = 2048;
    const int POSTTRIGGER_LEN = 2048;
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
    status = smSetSegIQCenterFreq(handle, 1.0e9);
    // Setup an rising edge video trigger at -40 dBm
    status = smSetSegIQVideoTrigger(handle, -40.0, smTriggerEdgeRising);
    // Setup a single segment capture that is video triggered
    status = smSetSegIQSegmentCount(handle, 1);
    status = smSetSegIQSegment(handle, 0, smTriggerTypeVideo, PRETRIGGER_LEN, POSTTRIGGER_LEN, 2.0);

    status = smConfigure(handle, smModeIQSegmentedCapture);
    if(status != smNoError) {
        // Unable to configure device
        const char *errStr = smGetErrorString(status);
        return;
    }

    // Get the maximum number of captures that can be buffered at once
    // Must be called after smConfigure
    int maxCaptures;
    smSegIQGetMaxCaptures(handle, &maxCaptures);

    // Allocate memory for the capture, 2 floats per I/Q sample returned
	std::vector<float> buf(CAPTURE_LEN * 2);

    // Start all the captures
    for(int i = 0; i < maxCaptures; i++) {
        smSegIQCaptureStart(handle, i);
    }

    // The captures will finish in order
    for(int i = 0; i < maxCaptures; i++) {
        // Block until complete
        status = smSegIQCaptureWait(handle, i);
        // Did the capture timeout?
        SmBool timedOut;
        status = smSegIQCaptureTimeout(handle, i, 0, &timedOut);
        // Read the I/Q data
        status = smSegIQCaptureRead(handle, i, 0, (float*)buf.data(), 0, buf.size() / 2);
        // Complete the capture
        status = smSegIQCaptureFinish(handle, i);
        
        // If desired, you could restart the capture here, maintaining the maximum
        //   buffered capture amount. For this example, we simply buffer the maximum
        //   amount and then finish them all once.
        //status = smSegIQCaptureStart(handle, i);
    }

    smAbort(handle);
    smCloseDevice(handle);
}
