/*
 *  Measure the speed of acquisition of VRT packets from a Signal Hound SM200C spectrum analyzer.
 *
 */

#include "sm_api.h"
#include "sm_api_vrt.h"

#pragma comment(lib, "sm_api")

#include "Windows.h"
#include <stdio.h>

void networkedSpeedTestVRT() {
	// Set up device
    int device = -1;
    SmStatus status = smOpenNetworkedDevice(&device, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT);
    if(status != smNoError) {
		printf("Could not open SM200C device");		
    }

    // Set IQ parameters
    smSetIQCenterFreq(device, 3.0e9);
    smSetIQSampleRate(device, 1);
    smSetIQBandwidth(device, smTrue, 20.0e6);
    smSetRefLevel(device, -20);

    // Set VRt parameters
    smSetVrtStreamID(device, 1);
    smSetVrtPacketSize(device, 32768);

    // Configure
    status = smConfigure(device, smModeIQStreaming); // VRT mode
    if(status != smNoError) {
		printf("Could not configure SM200C device");
    }

    // Allocate memory
    uint32_t contextWordCount;
    status = smGetVrtContextPktSize(device, &contextWordCount);
    if(status != smNoError) {
		printf("Could not get context packet size");
    }

	const int iterations = 100;
    const int dataPacketCount = 100;

    uint16_t samplesPerPkt;
    uint32_t dataWordCount;
    status = smGetVrtPacketSize(device, &samplesPerPkt, &dataWordCount);
    if(status != smNoError) {
		printf("Could not get data packet size and word count");
    }

    uint32_t wordCount = contextWordCount + dataWordCount * dataPacketCount;
    uint32_t *words = new uint32_t[wordCount];
    uint32_t *curr = words;

    // Get context packet
    uint32_t actualContextWordCount;
    status = smGetVrtContextPkt(device, curr, &actualContextWordCount);
    if(status != smNoError) {
		printf("Could not get context packet");
    }
    if(actualContextWordCount != contextWordCount) {
		printf("Context packet is not the expected size");
    }
    curr += contextWordCount;

    // Get data packets
    uint32_t actualDataWordCount;
		
	DWORD start = GetTickCount();
	for(int i = 0; i < iterations; i++) {
		status = smGetVrtPackets(device, curr, &actualDataWordCount, dataPacketCount, smFalse);
		if(status != smNoError) {
			printf("Could not get data packets");
		}
		if(actualDataWordCount != dataWordCount * dataPacketCount) {
			printf("Block of data packets is not the expected size");
		}
		
	}
	DWORD stop = GetTickCount();
	double duration = (double)(stop - start);

	double megasamplesPerSecond = (iterations * dataPacketCount * samplesPerPkt) / duration / 1e3;
	double durInSec = duration / 1e3;

	printf("%.2f MS/s in %.2f seconds\n\n", megasamplesPerSecond, durInSec);

	smCloseDevice(device);

	if(words) delete[] words;
}
