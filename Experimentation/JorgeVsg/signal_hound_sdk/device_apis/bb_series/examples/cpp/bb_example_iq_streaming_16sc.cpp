#include "bb_api.h"
#include <iostream>
#include <vector>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
In this example, we show streaming I/Q data as 16-bit complex ints and
show the conversion to amplitude corrected floats.
*/

void bbExampleIQStreaming16sc()
{
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << "Issue opening device\n";
        std::cout << bbGetErrorString(status) << "\n";
        return;
    }

    // Specify 16-bit complex shorts as the data type
    bbConfigureIQDataType(handle, bbDataType16sc);
    // Set center frequency
    bbConfigureIQCenter(handle, 2.4e9);
    // Set reference level 
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

    float iqCorrection = 0.0;
    bbGetIQCorrection(handle, &iqCorrection);

    // Allocate memory for BLOCK_SIZE complex values
    const int BLOCK_SIZE = 262144;

    // Allocate memory for both shorts and floats
    std::vector<int16_t> buffer16sc(BLOCK_SIZE * 2);
    std::vector<float> buffer32fc(BLOCK_SIZE * 2);

    // Perform capture
    status = bbGetIQUnpacked(handle, buffer16sc.data(), BLOCK_SIZE, 0, 0, 
        BB_TRUE, 0, 0, 0, 0);
    // Check status here

    // Convert to float, I/Q data is interleaved.
    // Apply the same logic to all re/im samples.
    for(int i = 0; i < buffer16sc.size(); i++) {
        // Convert to floats in the range [-1.0, 1.0]
        buffer32fc[i] = (((float)buffer16sc[i]) / 32768.0);
        // Apply correction.
        buffer32fc[i] *= iqCorrection;
    }

    // bbGetIQ can continue to be called to retrieve more IQ samples.

    // When done, stop streaming and close device.
    bbAbort(handle);
    bbCloseDevice(handle);
}
