#include "vrt_parser.h"

#include <cassert>

int main()
{
    // Set up device
    int device = -1;

	// Connect a USB or networked SM Series device
    SmStatus status = smOpenDevice(&device); // USB
	//SmStatus status = smOpenNetworkedDevice(&device, SM_ADDR_ANY, SM_DEFAULT_ADDR, SM_DEFAULT_PORT); // Networked

	if(status != smNoError) {
        // Could not open spec an
        assert(false, L"Could not open spec an.");
    }

    smSetIQCenterFreq(device, 3.0e9);
    smSetIQSampleRate(device, 1);
    smSetIQBandwidth(device, smTrue, 20.0e6);
    smSetRefLevel(device, -20);

    smSetVrtPacketSize(device, 32768);
    smSetVrtStreamID(device, 1);

    status = smConfigure(device, smModeIQStreaming); // VRT mode
    if(status != smNoError) {
        // Could not configure spec an
        assert(false, L"Could not configure spec an.");
    }

	//
    // Get blob of VRT data
	//

	int contextPacketCount = 10;
	int dataPacketsPerContextPacket = 10;
	int dataPacketCount = contextPacketCount * dataPacketsPerContextPacket;

    // Get context packet size
    uint32_t contextWordCount = 0;
    status = smGetVrtContextPktSize(device, &contextWordCount);
    if(status != smNoError) {
        // Could not get context packet size
        assert(false, L"Could not get context packet size.");
    }

    // Get data packet size
    uint32_t dataWordCount = 0;
    uint16_t samplesPerPacketReturn = 0;
    status = smGetVrtPacketSize(device, &samplesPerPacketReturn, &dataWordCount);
    if(status != smNoError) {
        // Could not get data packet size
        assert(false, L"Could not get data packet size.");
    }

    // Allocate memory for blob
    uint32_t wordCount = contextWordCount * contextPacketCount
					   + dataWordCount * dataPacketCount;
    uint32_t *words = new uint32_t[wordCount];
    uint32_t *curr = words;

    // Get VRT packets
    for(int i = 0; i < contextPacketCount; i++) {
        // Get context packet
        uint32_t actualContextWordCount;
        status = smGetVrtContextPkt(device, curr, &actualContextWordCount);
        if(status != smNoError) {
            // Could not get context packet
            assert(false, L"Could not get context packet.");
        }
        if(actualContextWordCount != contextWordCount) {
            // Context packet is not the expected size
            assert(false, L"Context packet is not the expected size.");
        }
        curr += actualContextWordCount;

        // Get data packets
        uint32_t actualDataWordCount;
        status = smGetVrtPackets(device, curr, &actualDataWordCount, dataPacketsPerContextPacket, smFalse);
        if(status != smNoError) {
            // Could not get data packets
            assert(false, L"Could not get data packets.");
        }
        if(actualDataWordCount != dataWordCount * dataPacketsPerContextPacket) {
            // Data packet is not the expected size
            assert(false, L"Data packet is not the expected size.");
        }
        curr += actualDataWordCount;
    }

    // Parse
    VRTParser parser;

    // Set a default reference level in the event that one or more data packets are parsed before the first context packet
    double reflevel;
    smGetRefLevel(device, &reflevel);
    parser.reflevel = reflevel;

    VRTUserDataPkt dataPkt;
    VRTUserContextPkt contextPkt;

    curr = words;
    while(curr - words < wordCount) {
        // Parse next packet
        SmVRTPacketType packetType;
        uint32_t packetSize;
        parser.Peek(curr, &packetType, &packetSize);

        switch(packetType) {
        case smVRTDataPacket:
            curr += parser.ParseDataPacket(curr, packetSize, dataPkt);
            break;
        case smVRTContextPacket:
            curr += parser.ParseContextPacket(curr, packetSize, contextPkt);
            break;
        default:
            // Memory pointed to is not a valid SM Series VRT packet
            break;
        }
    }

    int wordsParsed = curr - words;

	smCloseDevice(device);

	if(words) delete[] words;
}
