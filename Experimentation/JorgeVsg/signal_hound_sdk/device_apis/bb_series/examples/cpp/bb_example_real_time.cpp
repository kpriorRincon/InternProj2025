/** [realTimeExample1] */

#include "bb_api.h"
#include <iostream>
#include <vector>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example illustrates how to perform real-time spectrum analysis
with the API. The sweep and and persistence frames are retrieved.
*/

void bbExampleRealTime()
{
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << "Issue opening device\n";
        std::cout << bbGetErrorString(status) << "\n";
        return;
    }

    // Configure a 27MHz real-time stream at a 2.44GHz center
    bbConfigureRefLevel(handle, -20.0);
    bbConfigureCenterSpan(handle, 2.44e9, 20.0e6);
    bbConfigureSweepCoupling(handle, 10.e3, 10.0e3, 0.001, 
        BB_RBW_SHAPE_NUTTALL, BB_NO_SPUR_REJECT);
    bbConfigureAcquisition(handle, BB_MIN_AND_MAX, BB_LOG_SCALE);
    // Configure a frame rate of 30fps and 100dB scale for the frame
    bbConfigureRealTime(handle, 100.0, 30);

    // Configuration complete, initialize the device
    status = bbInitiate(handle, BB_REAL_TIME, 0);
    if(status < bbNoError) {
        std::cout << "Error configuring device\n";
        std::cout << bbGetErrorString(status);
        exit(-1);
    } 

    // Get sweep characteristics and allocate memory for sweep and
    // real-time frames.
    uint32_t sweepSize;
    double binSize, startFreq;
    bbQueryTraceInfo(handle, &sweepSize, &binSize, &startFreq);
    int frameWidth, frameHeight;
    bbQueryRealTimeInfo(handle, &frameWidth, &frameHeight);

    std::vector<float> sweep(sweepSize);
    std::vector<float> frame(frameWidth * frameHeight);
    std::vector<float> alphaFrame(frameWidth * frameHeight);

    // Retrieve roughly 1 second worth of real-time persistence frames and sweeps.
    // Ignore min sweep, pass NULL as parameter.
    int frameCount = 0;
    while(frameCount++ < 30) {
        bbFetchRealTimeFrame(handle, nullptr, sweep.data(), frame.data(), alphaFrame.data());
    }

    // Finished/close device
    bbAbort(handle);
    bbCloseDevice(handle);
}

/** [realTimeExample1] */
