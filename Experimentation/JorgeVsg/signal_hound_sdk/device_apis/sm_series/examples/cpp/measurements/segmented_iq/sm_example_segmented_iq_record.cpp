#include "sm_api.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

// SM200B/SM435B only.
// This example illustrates capturing a full 2 second I/Q capture at the full
//   160MHz I/Q bandwidth and storing it to a binary file.
// This example illustrates retrieving the data as 16-bit shorts, pulling
//   data off the device in chunks as opposed to all at once which would require
//   a 2GB contiguous buffer, and storing it sequentially to disk using the 
//   FILE C interface.

void sm_example_segmented_iq_record()
{
    // Number of I/Q samples, 2 seconds worth
    const int CAPTURE_LEN = 500e6;

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
    status = smSetSegIQDataType(handle, smDataType16sc);
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

    // 2 floats for each I/Q sample
    const int bufSize = 1e6;
	std::vector<short> buf(2 * bufSize);

    // Start the measurement
    status = smSegIQCaptureStart(handle, 0);
    // Block until complete
    status = smSegIQCaptureWait(handle, 0);

    // Read through the buffer and read all the I/Q, and store it to disk
    FILE *f;
    fopen_s(&f, "iq.bin", "wb");
    int samplesRead = 0;

    // Loop through the full buffer reading at most bufSize worth of I/Q
    //   samples on each iteration.
    while(samplesRead < CAPTURE_LEN) {
        int samplesLeftToRead = (CAPTURE_LEN - samplesRead);
        int samplesToRead = (samplesLeftToRead > bufSize) ? bufSize : samplesLeftToRead;

        status = smSegIQCaptureRead(handle, 0, 0, &buf[0], samplesRead, samplesToRead);

        int bytesToWrite = sizeof(short) * 2 * samplesToRead;
        size_t res = fwrite(&buf[0], bytesToWrite, 1, f);
        assert(res == 1);

        samplesRead += samplesToRead;
    }

    status = smSegIQCaptureFinish(handle, 0);

    fclose(f);

    smAbort(handle);
    smCloseDevice(handle);
}
