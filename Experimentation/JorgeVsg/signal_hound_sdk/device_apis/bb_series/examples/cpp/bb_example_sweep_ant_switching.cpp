#include "bb_api.h"

#include <iostream>
#include <vector>

void bbExampleSweepAntSwitching()
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
        std::cout << "Not a BB60D, does not support sweep antenna switching\n";
        bbCloseDevice(handle);
        exit(-1);
    }

    // Set 1MHz UART baud rate
    bbSetUARTRate(handle, BB60D_UART_BAUD_1000K);

    // Set UART states/counts
    const int states = 6;
    uint8_t data[states];
    double freqs[states];
    for(int i = 0; i < states; i++) {
        data[i] = i;
        freqs[i] = i * 1.0e9;
    }
    bbEnableUARTSweeping(handle, freqs, data, states);

    // Configure the IO for UART output
    bbConfigureIO(handle, BB60D_PORT1_DISABLED, BB60D_PORT2_OUT_UART);

    // Now configure the device for a full band sweep
    double start = 100.0e3;
    double stop = 6.0e9;
    bbConfigureRefLevel(handle, 0.0);
    bbConfigureCenterSpan(handle, (stop + start) / 2.0, stop - start);
    bbConfigureSweepCoupling(handle, 100.0e3, 100.0e3, 0.001, BB_RBW_SHAPE_FLATTOP, BB_FALSE);
    bbConfigureAcquisition(handle, BB_MIN_AND_MAX, BB_LOG_SCALE);
    bbConfigureProcUnits(handle, BB_POWER);

    // Initiate the I/Q stream, UART switching has begun when this function returns
    status = bbInitiate(handle, BB_SWEEPING, 0);
    if(status < bbNoError) {
        std::cout << "Error configuring device\n";
        std::cout << bbGetErrorString(status);
        exit(-1);
    } 

    // Allocate memory for sweep
    uint32_t sweepLen;
    double binSize, startFreq;
    bbQueryTraceInfo(handle, &sweepLen, &binSize, &startFreq);

    std::vector<float> sweepMin(sweepLen);
    std::vector<float> sweepMax(sweepLen);

    for(int i = 0; i < 1000; i++) {
        status = bbFetchTrace_32f(handle, sweepLen, sweepMin.data(), sweepMax.data());
        if(status < bbNoError) {
            std::cout << "Error getting sweep\n";
            std::cout << bbGetErrorString(status);
            exit(-1);
        } 
        std::cout << "Sweep " << i << std::endl;
    }

    bbAbort(handle);
    bbCloseDevice(handle);
}