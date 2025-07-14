#include "bb_api.h"
#include <iostream>
#include <vector>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/* 
This example illustrates how to set up the BB60D to output UART bytes while I/Q streaming
for the purposes of antenna switching and performing pseudo doppler.
This example does not include the actual pseudo doppler math.
*/

const int TRIGGER_SENTINEL = -1;

void bbExamplePseudoDoppler()
{
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << "Issue opening device\n";
        std::cout << bbGetErrorString(status) << "\n";
        return;
    }

    int deviceType;
    bbGetDeviceType(handle, &deviceType);
    if(deviceType != BB_DEVICE_BB60D) {
        std::cout << "Not a BB60D, does not support pseudo-doppler antenna switching\n";
        bbCloseDevice(handle);
        exit(-1);
    }

    // Set 1MHz UART baud rate
    bbSetUARTRate(handle, BB60D_UART_BAUD_1000K);

    // Set UART states/counts
    const int states = 4;
    uint8_t data[states];
    uint32_t counts[states];
    for(int i = 0; i < states; i++) {
        data[i] = i + 8; // UART data to write
        counts[i] = 10000; // # of 40MHz counts at this state
    }
    bbEnableUARTStreaming(handle, data, counts, states);

    // Configure the IO for UART output
    bbConfigureIO(handle, BB60D_PORT1_DISABLED, BB60D_PORT2_OUT_UART);

    // Now configure the device for I/Q streaming
    bbConfigureRefLevel(handle, 0.0);
    bbConfigureIQCenter(handle, 1.0e9);
    bbConfigureIQ(handle, 1, 27.0e6);
    bbConfigureIQDataType(handle, bbDataType32fc);
    bbConfigureIQTriggerSentinel(TRIGGER_SENTINEL);

    // Initiate the I/Q stream, UART switching has begun when this function returns
    status = bbInitiate(handle, BB_STREAMING, BB_STREAM_IQ);
    if(status < bbNoError) {
        std::cout << "Error configuring device\n";
        std::cout << bbGetErrorString(status);
        exit(-1);
    } 

    // Allocate memory for I/Q retrieval
    int len = 1e6;
    std::vector<float> iq(len * 2);
    const int trigBufSize = 200;
    int triggers[trigBufSize];

    int acq = 0;
    while(acq++ < (1*40)) {
        status = bbGetIQUnpacked(handle, iq.data(), len, triggers, trigBufSize, BB_FALSE, 0, 0, 0, 0);
        if(status < bbNoError) {
            std::cout << "Error getting I/Q data\n";
            std::cout << bbGetErrorString(status) << "\n";
            bbCloseDevice(handle);
            exit(-1);
        }    

        int triggerCount = 0;
        while(triggers[triggerCount] != TRIGGER_SENTINEL) {
            triggerCount++;
            if(triggerCount >= trigBufSize) {
                break;
            }
        }

        std::cout << "Trigger count " << triggerCount << "\n";

        if(triggerCount >= trigBufSize) {
            std::cout << "Too many triggers\n";
        }

        for(int i = 0; i < triggerCount - 1; i++) {
            int delta = triggers[i+1] - triggers[i];
            double deltaTime = (1.0 / 40.0e6) * delta;
            //if(i == 0) {
                std::cout << "Trigger delta " << (deltaTime * 1.0e3) << "ms\n";
            //}
            if(deltaTime < 0.0001) {
                std::cout << "Bad delta\n";
                int bp = 0;
            }
        }
    }

    bbAbort(handle);
    bbCloseDevice(handle);
}