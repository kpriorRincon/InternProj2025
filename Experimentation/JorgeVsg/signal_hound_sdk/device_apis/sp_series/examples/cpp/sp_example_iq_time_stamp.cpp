#include <cstdio>
#include <cstdlib>
#include <vector>

#include <Windows.h>

#include "sp_api.h"

// This example demonstrates how to configure a device for GPS time stamped
//   I/Q acquisitions. The GPS antenna should be connected to the device.
// Once lock and discipline is established, prints GPS timing information in a loop.

void sp_example_iq_time_stamp()
{
    int handle = -1;
    SpStatus status = spNoError;

    // Open device
    status = spOpenDevice(&handle);

    if(status != spNoError) {
        printf("Unable to open device\n");
        exit(-1);
    }

    // Configure the receiver for I/Q acquisition
    // 900MHz center freq, 50MS/s sample rate, no additional software filtering, floats
    spSetIQCenterFreq(handle, 900.0e6); 
    spSetIQSampleRate(handle, 1);
    spSetIQBandwidth(handle, 40.0e6);
    spSetIQDataType(handle, spDataType32fc);

    // This function enables GPS disciplining. When enabled and when the internal
    //   GPS achieves lock, the GPS 1PPS signal will discpline the 10MHz timebase 
    //   of the SP device. 
    spSetGPSTimebaseUpdate(handle, spTrue);

    // Initialize the receiver with the above settings
    status = spConfigure(handle, spModeIQStreaming);
    if(status != spNoError) {
        printf("Unable to configure device\n");
        printf("%s\n", spGetErrorString(status));
        spCloseDevice(handle);
        exit(-1);
    }

    // Query the receiver I/Q stream characteristics
    // Should match what we set earlier
    double actualSampleRate, actualBandwidth;
    spGetIQParameters(handle, &actualSampleRate, &actualBandwidth);

    // Wait for GPS discipline
    // This can take ~1 minute in the best scenario (previously disciplined from an
    //   earlier invocation of the program) to several minutes or more from a cold start.
    // Waiting for discipline is not required. Technically as soon as the GPS state is
    //   reported as locked, the API reports GPS timestamps, without proper disciplining
    //   the system clock will drift relative to the GPS clock.
    int waitIter = 0;
    SpGPSState gpsState;
    do {
        spGetGPSState(handle, &gpsState);
        Sleep(1000);
        printf("Wait for GPS, %d second\n", ++waitIter);
    } while(gpsState != spGPSStateDisciplined);

    printf("Device now disciplined\n");

    // Now that the GPS is disciplined, let's start streaming I/Q and timestamps

    // Will request 100 1/100th size buffers, and will print timestamp once per
    //   100 requests. 500k samples per request.
    int bufferLength = 500000;
    // Times 2 for I and Q interleaved values
    std::vector<float> iqBuf(bufferLength * 2);

    // How many seconds will we print timestamp information
    int timestampsToPrint = 2000;

    // Initialize the starting time and set purge to true to flush any I/Q
    //   that accumulated while waiting for the GPS discipline flag to set.
    // Starting time in nanos since epoch.
    int64_t startingTime;
    spGetIQ(handle, &iqBuf[0], bufferLength, 0, 0, &startingTime, spTrue, 0, 0);

    while(timestampsToPrint > 0) {
        int64_t nsSinceEpoch;

        // Collect 1 second of I/Q data in loop
        for(int i = 0; i < 100; i ++) {
            // One I/Q data request
            spGetIQ(handle, &iqBuf[0], bufferLength, 0, 0, &nsSinceEpoch, spFalse, 0, 0);

            // After the spGetIQ function returns, the timestamp (nsSinceEpoch) will 
            //  represent the time of the first sample in the returned I/Q

            // Do something with data...
        }

        // Print several metrics each second
        double secSinceEpoch = (double)nsSinceEpoch / 1.0e9;
        int64_t deltaNanos = nsSinceEpoch - startingTime;
        double deltaSeconds = (double)deltaNanos / 1.0e9;
        printf("Seconds since epoch %.0f\n", secSinceEpoch);
        printf("Nanos since epoch %lld\n", nsSinceEpoch);
        printf("Seconds elapsed %.9f\n", deltaSeconds);
        printf("Nanos elapsed %lld\n", deltaNanos);
        printf("\n");

        timestampsToPrint--;
    }

    // Finished
    spCloseDevice(handle);
}