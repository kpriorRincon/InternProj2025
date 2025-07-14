/** [vrtExample1] */

/*
 *  Get a VRT Context packet from a Signal Hound SM Series device followed by a block of 1000 Signal Data packets
 *
 */

#include "sm_api.h"
#include "sm_api_vrt.h"

#pragma comment(lib, "sm_api")

void getVRTPackets() {
	// Set up device
    int device = -1;
    //SmStatus status = smOpenDevice(&device); // USB
    SmStatus status = smOpenNetworkedDevice(&device, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT); // Networked
    if(status != smNoError) {
        // Could not open Sm Series device
    }

    // Set IQ parameters
    smSetIQCenterFreq(device, 3.0e9);
    smSetIQSampleRate(device, 2);
    smSetIQBandwidth(device, smTrue, 20.0e6);
    smSetRefLevel(device, -20);

    // Set VRt parameters
    smSetVrtStreamID(device, 1);
    smSetVrtPacketSize(device, 16384);

    // Configure
    status = smConfigure(device, smModeIQStreaming); // VRT mode
    if(status != smNoError) {
        // Could not configure SM Series device
    }

    // Allocate memory
    uint32_t contextWordCount;
    status = smGetVrtContextPktSize(device, &contextWordCount);
    if(status != smNoError) {
        // Could not get context packet size
    }

    const int dataPacketCount = 1000;

    uint16_t samplesPerPkt;
    uint32_t dataWordCount;
    status = smGetVrtPacketSize(device, &samplesPerPkt, &dataWordCount);
    if(status != smNoError) {
        // Could not get data packet size and word count
    }

    uint32_t wordCount = contextWordCount + dataWordCount * dataPacketCount;
    uint32_t *words = new uint32_t[wordCount];
    uint32_t *curr = words;

    // Get context packet
    uint32_t actualContextWordCount;
    status = smGetVrtContextPkt(device, curr, &actualContextWordCount);
    if(status != smNoError) {
        // Could not get context packet
    }
    if(actualContextWordCount != contextWordCount) {
        // Context packet is not the expected size
    }
    curr += contextWordCount;

    // Get data packets
    uint32_t actualDataWordCount;
    status = smGetVrtPackets(device, curr, &actualDataWordCount, dataPacketCount, smFalse);
    if(status != smNoError) {
        // Could not get data packets
    }
    if(actualDataWordCount != dataWordCount * dataPacketCount) {
        // Block of data packets is not the expected size
    }

	smCloseDevice(device);

	if(words) delete[] words;
}

/** [vrtExample1] */
