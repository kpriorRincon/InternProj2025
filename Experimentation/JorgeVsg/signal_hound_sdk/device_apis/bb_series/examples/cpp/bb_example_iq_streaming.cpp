/** [iqStreamingExample1] */

#include "bb_api.h"
#include <iostream>
#include <vector>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example demonstrates how to configure, initialize, and retrieve data
in I/Q streaming mode. I/Q streaming mode provides continuous I/Q data
at a fixed center frequency with a selectable sample rate.
*/

void bbExampleIQStreaming()
{
    // Connect device
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << "Issue opening device\n";
        std::cout << bbGetErrorString(status) << "\n";
        return;
    }

    // Configure the measurement parameters

    // Specify 32-bit floating point complex values
    bbConfigureIQDataType(handle, bbDataType32fc);
    // Set center frequency
    bbConfigureIQCenter(handle, 1.0e9);
    // Set reference level, maximum expected input amplitude
    bbConfigureRefLevel(handle, -20.0);
    // Set a sample rate of 40MS/s and a bandwidth of 27MHz
    bbConfigureIQ(handle, 1, 27.0e6);

    // Initiate the device, once this function returns the device
    // will be streaming I/Q.
    status = bbInitiate(handle, BB_STREAMING, BB_STREAM_IQ);
    if(status != bbNoError) {
        std::cout << "Initiate error\n";
        std::cout << bbGetErrorString(status) << "\n";
        bbCloseDevice(handle);
        return;
    }

    // Get I/Q stream characteristics
    double sampleRate, bandwidth;
    bbQueryIQParameters(handle, &sampleRate, &bandwidth);

    // Allocate memory for BLOCK_SIZE number of complex values
    // This is the number of I/Q samples we will request each function call.
    const int BLOCK_SIZE = 262144;
    std::vector<float> buffer(BLOCK_SIZE * 2);

    // Perform the capture, pass NULL for any parameters we don't care about
    status = bbGetIQUnpacked(handle, buffer.data(), BLOCK_SIZE, 0, 0, 
        BB_FALSE, 0, 0, 0, 0);
    // Check status here

    // At this point, BLOCK_SIZE IQ data samples have been retrieved and
    //   stored in the buffer array. Any processing on the data should happen here.

    // Continue to call bbGetIQ with the purge flag set to false, which ensures
    //   I/Q data is continuous from the last call.
    for(int i = 0; i < 10; i++) {
        status = bbGetIQUnpacked(handle, buffer.data(), BLOCK_SIZE, 0, 0, 
            BB_FALSE, 0, 0, 0, 0);
        // Check status here

        // Do processing
    }

    // When done, stop streaming and close device.
    bbAbort(handle);
    bbCloseDevice(handle);
}

/** [iqStreamingExample1] */
