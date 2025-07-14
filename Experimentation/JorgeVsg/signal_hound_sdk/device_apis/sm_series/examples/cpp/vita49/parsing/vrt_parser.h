#ifndef PARSER_H
#define PARSER_H

#include "sm_api.h"
#include "sm_api_vrt.h"
#include "sh_vrt.h"

#pragma comment(lib, "sm_api")

class VRTParser
{
public:
    int ParseDataPacket(uint32_t *words, uint32_t wordCount, VRTUserDataPkt &parsed);
    int ParseContextPacket(uint32_t *words, uint32_t wordCount, VRTUserContextPkt &parsed);
    void Peek(const uint32_t *pkts, SmVRTPacketType *packetType, uint32_t *packetSize);

    VRTUserPktPrologue ParsePrologue(const VRTPktPrologue &prologue);
    void UnpackSamples(const int16_t* src, float* dst, int len);

    double reflevel;
};

#endif // PARSER_H
