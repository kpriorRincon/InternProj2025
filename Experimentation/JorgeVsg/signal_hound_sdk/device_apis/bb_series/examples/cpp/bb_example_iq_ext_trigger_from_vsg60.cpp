#include "bb_api.h"

#include "vsg_api.h"

#include <complex>
#include <iostream>
#include <vector>
#include <thread>
#include <cmath>

#ifdef _WIN32
#include "Windows.h"

#pragma comment(lib, "bb_api.lib")
#pragma comment(lib, "vsg_api.lib")
#endif

/*
This example demonstrates sending a trigger and pulsed signal from a Signal Hound VSG60A
vector signal generator, and using the BB60 to wait for the trigger and receive the signal.
*/

struct Cplx32f { 
    float re, im; 
};

const double pulseWidth = 1.0e-6; // 1us
const double pulsePeriod = 10.0e-6; // 10us

const int sampleOffset = 10;

const int triggersPerSecond = 1000;
const int transmitTimeSec = 3;

const double freq = 1.0e9; // 1GHz
const double level = -10.0; // dBm

const double sampleRate = 50.0e6; // samples per second

std::thread receiveThread;

int bbHandle;

void waitForTriggers()
{
	// This is how much data we want after the external trigger
    const int N = 1e3;

    // I/Q capture buffer
    std::vector<std::complex<float>> buffer(N);    
	
	const int triggersArrayLen = 1000;
	int triggers[triggersArrayLen];

	const int postTriggerSampleWindow = 50;

	double avgOffset = 0;

	int triggerCount = 0;
	while(triggerCount < triggersPerSecond * transmitTimeSec) {
        // Perform the capture, pass NULL for any parameters we don't care about
        bbStatus status = bbGetIQUnpacked(bbHandle, &buffer[0], N, triggers, triggersArrayLen, BB_FALSE, 0, 0, 0, 0);

        // At this point, N I/Q samples have been retrieved and
        //  stored in the buffer array. If any triggers were seen during the capture
        //  of the returned samples, the trigger array will contain indexes into
        //  the buffer array where each trigger was seen.

        // A trigger value of 0 indicates no trigger.

		if(triggers[0] == 0) {
			continue;
			// Assumed that zero in the first array position of triggers means no triggers were found, 
			//   even though it could mean that a trigger was seen in the first array position of the capture
		}

		for(int i = 0; i < triggersArrayLen; i++) {
			if(triggers[i] == 0) break;

			int offset = 0;
			for(int j = triggers[i]; j < triggers[i] + postTriggerSampleWindow; j++) {
				if(j >= buffer.size()) continue;

				double ampl = 10 * log10(buffer[j].real() * buffer[j].real() + buffer[j].imag() * buffer[j].imag());
				if(ampl >= (level - 10)) {
					break;
				}
				offset++;
			}

			offset -= sampleOffset;
			avgOffset += offset;
			triggerCount++;

			printf("Trigger %d: %d samples offset\n", triggerCount, offset);
		}
    }

	avgOffset /= triggerCount;

	printf("%d triggers seen.\nAverage sample offset between trigger and pulse rising edge: %.2f samples\n", triggerCount, avgOffset);
}


void bbExampleIQExtTriggerWithVSG60()
{
	// Open devices

	// Open BB60
    bbStatus bbStat = bbOpenDevice(&bbHandle);
    if(bbStat != bbNoError) {
        std::cout << "Issue opening BB60 device\n";
        std::cout << bbGetErrorString(bbStat) << "\n";
        return;
    }

	// Open VSG60
	int vsgHandle;
	VsgStatus vsgStatus = vsgOpenDevice(&vsgHandle);
	if(vsgStatus != vsgNoError) {
        std::cout << "Issue opening VSG60 device\n";
		std::cout << vsgGetErrorString(vsgStatus) << "\n";
        return;
    }

	// Configure devices
	
	// Configure VSG60 to send triggers and output pulsed signal	
    vsgSetFrequency(vsgHandle, freq);
    vsgSetLevel(vsgHandle, level);
    vsgSetSampleRate(vsgHandle, sampleRate);

    // Create pulse with specified width and period
    const Cplx32f cplxOne = {1.0, 0.0}, cplxZero = {0.0, 0.0};
    std::vector<Cplx32f> iq; // The I/Q waveform

    const int pulseOnSamples = pulseWidth * sampleRate;
    const int pulseOffSamples = (pulsePeriod - pulseWidth) * sampleRate;

	for(int j = 0; j < sampleOffset; j++) {
		iq.push_back(cplxZero);
	}			
	for(int j = 0; j < pulseOnSamples; j++) {
		iq.push_back(cplxOne);
	}
	for(int j = 0; j < pulseOffSamples; j++) {
		iq.push_back(cplxZero);
	}

	// Configure BB60 measurement parameters
    bbConfigureIQDataType(bbHandle, bbDataType32fc); // 32-bit floating point complex values
    bbConfigureIQCenter(bbHandle, freq);
    bbConfigureRefLevel(bbHandle, level + 5);
    bbConfigureIQ(bbHandle, 1, 27.0e6);

	// Configure BNC port 2 for rising edge trigger detection
	int deviceType;
    bbGetDeviceType(bbHandle, &deviceType);
    if(deviceType == BB_DEVICE_BB60D) {
        // BB60D
        bbConfigureIO(bbHandle, BB60D_PORT1_DISABLED, BB60D_PORT2_IN_TRIG_RISING_EDGE);
    } else {
        // BB60C/A devices
        bbConfigureIO(bbHandle, 0, BB60C_PORT2_IN_TRIG_RISING_EDGE);
    }

    // Initiate the BB60
	// Once this function returns it will be streaming I/Q
    bbStat = bbInitiate(bbHandle, BB_STREAMING, BB_STREAM_IQ);
    if(bbStat != bbNoError) {
        std::cout << "Initiate error\n";
        std::cout << bbGetErrorString(bbStat) << "\n";
        bbCloseDevice(bbHandle);
        return;
    }

	// Spin off a thread for the BB60 to wait for triggers
	receiveThread = std::thread(waitForTriggers);

	Sleep(1000);

	// Transmit our pulsed waveform for a few seconds at the specified rate
    const double sleepTimeMS = 1000.0 / triggersPerSecond;
    for(int i = 0; i < transmitTimeSec + 1; i++) { // Transmit for an extra second to give a margin of error to receive thread
		for(int j = 0; j < triggersPerSecond; j++) {
			vsgSubmitTrigger(vsgHandle);
			vsgSubmitIQ(vsgHandle, (float*)&iq[0], iq.size());
			
			Sleep(sleepTimeMS);
		}
    }

    vsgFlushAndWait(vsgHandle);

	// Sleep while BB60C receives
	Sleep(transmitTimeSec * 1000 + 2000); // Pad a couple seconds

    // Done, stop streaming and close devices
	vsgAbort(vsgHandle);
    vsgCloseDevice(vsgHandle);

    bbAbort(bbHandle);
    bbCloseDevice(bbHandle);

	if(receiveThread.joinable()) {
		receiveThread.join();
	}
}
