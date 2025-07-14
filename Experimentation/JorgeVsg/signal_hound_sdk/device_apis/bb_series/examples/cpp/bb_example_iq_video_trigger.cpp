#include "bb_api.h"
#include <iostream>
#include <vector>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example shows how to perform video triggering on I/Q data from the BB60.
A very simple threshold detector is used. The example waits for 1 trigger event,
captures a specific amount of I/Q data after the trigger, then quits.
*/

void bbExampleIQVideoTrigger()
{
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << "Issue opening device\n";
        std::cout << bbGetErrorString(status) << "\n";
        return;
    }

    // Now configure the measurement parameters

    // We want 32-bit floating point complex values
    bbConfigureIQDataType(handle, bbDataType32fc);
    // Set center frequency to 1GHz
    bbConfigureIQCenter(handle, 1.0e9);
    // Set reference level 
    bbConfigureRefLevel(handle, -20.0);
    // Set a sample rate of 40MS/s / 64 = 625kS/s, and a BW of 500kHz
    bbConfigureIQ(handle, 64, 500.0e6);

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

    // Our trigger parameters, convert trigger level to linear power so that
    //   the threshold can be tested without calculating logarithms.
    const double thresholdDBM = -40.0;
    const double thresholdPow = pow(10.0, thresholdDBM / 10.0);
    // How many samples after the trigger we care about, to make the logic easy,
    //   it is shorter than our I/Q request length.
    const int triggerLen = 20000;

    // Allocate memory for IQ_REQUEST_LEN number of complex values
    const int IQ_REQUEST_LEN = 32768;
    std::vector<float> buffer(IQ_REQUEST_LEN * 2);

    // If we find a trigger, this value will be set to the index of that trigger position.
    int triggerPos = 0;

    // Now look for trigger
    while(true) {
        // Perform the capture, pass NULL for any parameters we don't care about
        status = bbGetIQUnpacked(handle, buffer.data(), IQ_REQUEST_LEN, 
            0, 0, BB_FALSE, 0, 0, 0, 0);

        bool triggerFound = false;

        // Loop through all I/Q samples looking for a sample that exceeds the threshold.
        // This can be expensive at higher sample rate.
        for(int i = 0; i < IQ_REQUEST_LEN; i++) {
            // i*i + q*q
            double levelPow = buffer[i*2]*buffer[i*2] + buffer[i*2+1]*buffer[i*2+1];
            if(levelPow > thresholdPow) {
                triggerPos = i;
                triggerFound = true;
                break;
            }
        }

        if(triggerFound) {
            break;
        }
    }

    // We have found our trigger, lets collect the first "triggerLen" samples after
    //   the trigger.
    // Allocate buffer for triggered capture
    std::vector<float> triggeredBuf(triggerLen * 2);
    // Calculate how many samples after the trigger are in the current capture
    int samplesRemainingInCapture = IQ_REQUEST_LEN - triggerPos;
    // Copy as many as we can up to the trigger length.
    int samplesToCopy = (samplesRemainingInCapture > triggerLen) ? triggerLen : samplesRemainingInCapture;
    memcpy(&triggeredBuf[0], &buffer[triggerPos*2], sizeof(float) * 2 * samplesToCopy);

    // Depending on where the trigger is, we may not have enough samples for our capture
    // Grab the remaining needed samples, and append them to what we have.
    if(samplesToCopy < triggerLen) {
        int samplesLeft = triggerLen - samplesToCopy;
        status = bbGetIQUnpacked(handle, &triggeredBuf[samplesToCopy*2], 
            samplesLeft, 0, 0, BB_FALSE, 0, 0, 0, 0);
    }

    // When done, stop streaming and close device.
    bbAbort(handle);
    bbCloseDevice(handle);
}