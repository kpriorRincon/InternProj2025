#include "bb_api.h"

#define _CRT_SECURE_NO_WARNINGS
#include <cassert>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <sys/types.h>
#include <sys/timeb.h>
#include <Windows.h>
#include <vector>

#pragma comment(lib,"bb_api.lib")

/*
This example illustrates how to configure the API for I/Q timestamping.
Timestamping through the API is currently only available on Windows platforms.
This code requires a GPS device that transmits NMEA data via COM port to the PC
  and has a 1PPS output.
For additional accuracy, a 10MHz timebase can be used to discipline the BB60 sample clocks.
*/

void bbExampleIQTimestamping()
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

    // Configure the device to accept input triggers on port 2 
    // The 1 PPS trigger will be connected to port 2 
    // Configuration dependent on device type
    if(deviceType == BB_DEVICE_BB60D) {
        // BB60D
        bbConfigureIO(handle, BB60D_PORT1_DISABLED, BB60D_PORT2_IN_TRIG_RISING_EDGE);
    } else {
        // BB60C/A
        bbConfigureIO(handle, 0, BB60C_PORT2_IN_TRIG_FALLING_EDGE);
    }

    // Configure an I/Q data stream as usual 
    bbConfigureIQCenter(handle, 2400.0e6); 
    bbConfigureRefLevel(handle, -20.0); 
    bbConfigureIQ(handle, 4, 6.0e6);

    // At this point the GPS receiver must be operational 
    // The RS232 connection cannot be open, and the COM port 
    // and the baud rate must be known 
    // Ensure the receiver is locked 
    if(bbSyncCPUtoGPS(3, 38400) != bbNoError) {
        printf("Error configuring COM port");
        assert(false);
        return;
    }

    // If syncCPUtoGPS returned successfully the device can now be initialized 
    // and the RS232 connection should now be closed. 
    // Note: BB_TIME_STAMP is required so the device treats input triggers as the 
    // GPS 1 PPS 
    bbInitiate(handle, BB_STREAMING, BB_STREAM_IQ | BB_TIME_STAMP);

    // Verify bandwidth and sample rate
    double bandwidth;
    int sampleRate;
    bbQueryStreamInfo(handle, 0, &bandwidth, &sampleRate);

    // Allocate memory for IQ values
    const int BLOCK_SIZE = 32768;
    std::vector<float> buffer(BLOCK_SIZE * 2);

    // Wait for first PPS trigger to occur in the API
    // Until the first PPS trigger arrives at the device, it is not possible for the IQ data time
    //  stamps to be accurate. 
    Sleep(1500);

    int numSecsPassed = 0;
    int lastSec = 0;
    while(numSecsPassed < 20) {
        int sec, ns;
        bbGetIQUnpacked(handle, buffer.data(), BLOCK_SIZE, 0, 0, 
            BB_TRUE, 0, 0, &sec, &ns);

        if(sec != lastSec) {
            lastSec = sec;
            numSecsPassed++;
            printf("Seconds since epoch: %d.%d\n", sec, ns);

            time_t t = sec;
            tm tm;
            localtime_s(&tm, &t);
            char timeStr[64];
            asctime_s(timeStr, &tm);
            printf("Date/Time: %s\n", timeStr);
        }
    }

    // Done 
    bbCloseDevice(handle);
}