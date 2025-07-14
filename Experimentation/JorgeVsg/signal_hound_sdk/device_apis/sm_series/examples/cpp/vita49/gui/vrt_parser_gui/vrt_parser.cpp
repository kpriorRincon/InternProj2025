#include "vrt_parser.h"

#include <cmath>

#define VRT_DATA_PACKET_CODE		1
#define VRT_CONTEXT_PACKET_CODE		4

int VRTParser::ParseDataPacket(uint32_t *words, uint32_t wordCount, VRTUserDataPkt &parsed)
{
    vrtSwapBytes(words, wordCount);

    uint32_t *base = words;
    uint32_t *curr = base;

    // Copy metadata
    VRTDataPktMetadata metadata;
    memcpy(&metadata, curr, sizeof(VRTDataPktMetadata));
    curr += sizeof(VRTDataPktMetadata) / sizeof(uint32_t);

    // Parse prologue
    parsed.prologue = ParsePrologue(metadata.prologue);

    // Parse data payload
    int iqBufSize = parsed.prologue.header.packetSize - sizeof(VRTDataPktMetadata)/sizeof(uint32_t) - 1; // Trailer is one word
    std::vector<int16_t> intBuf(iqBufSize * 2);
    parsed.data.resize(iqBufSize * 2);
    memcpy(&intBuf[0], curr, iqBufSize * sizeof(uint32_t));
    UnpackSamples(&intBuf[0], &parsed.data[0], iqBufSize);
    curr += iqBufSize;

    // Parse Trailer
    uint32_t trailer;
    memcpy(&trailer, curr, sizeof(trailer));
    curr += sizeof(trailer) / sizeof(uint32_t);

    // Calibrated Time Indicator
    if(vrtGetBit(trailer, VRT_TIME_ENABLE)) {
        parsed.trailer.isCalibratedTime.enabled = true;
        if(vrtGetBit(trailer, VRT_TIME_INDICATOR)) {
            parsed.trailer.isCalibratedTime.indicator = true;
        }
    }
    // Valid Data Indicator
    if(vrtGetBit(trailer, VRT_VALID_DATA_ENABLE)) {
        parsed.trailer.isValidData.enabled = true;
        if(vrtGetBit(trailer, VRT_VALID_DATA_INDICATOR)) {
            parsed.trailer.isValidData.indicator = true;
        }
    }
    // Reference Lock Indicator
    if(vrtGetBit(trailer, VRT_REF_LOCK_ENABLE)) {
        parsed.trailer.isReferenceLock.enabled = true;
        if(vrtGetBit(trailer, VRT_REF_LOCK_INDICATOR)) {
            parsed.trailer.isReferenceLock.indicator = true;
        }
    }
    // Over-Range Indicator
    if(vrtGetBit(trailer, VRT_OVER_RANGE_ENABLE)) {
        parsed.trailer.isOverRange.enabled = true;
        if(vrtGetBit(trailer, VRT_OVER_RANGE_INDICATOR)) {
            parsed.trailer.isOverRange.indicator = true;
        }
    }
    // Sample Loss Indicator
    if(vrtGetBit(trailer, VRT_SAMPLE_LOSS_ENABLE)) {
        parsed.trailer.isSampleLoss.enabled = true;
        if(vrtGetBit(trailer, VRT_SAMPLE_LOSS_INDICATOR)) {
            parsed.trailer.isSampleLoss.indicator = true;
        }
    }
    // Associated Context Packet Count
    if(vrtGetBit(trailer, VRT_DATA_TRAILER_E_BIT)) {
        parsed.trailer.associatedContextPktCount = vrtGetAssociatedContextPktCount(trailer);

    }
    return curr - base;
}

int VRTParser::ParseContextPacket(uint32_t *words, uint32_t wordCount, VRTUserContextPkt &parsed)
{
    vrtSwapBytes(words, wordCount);

    uint32_t *base = words;
    uint32_t *curr = base;

    // Copy metadata
    VRTContextPktMetadata metadata;
    memcpy(&metadata, curr, sizeof(VRTContextPktMetadata));
    curr += sizeof(VRTContextPktMetadata) / sizeof(uint32_t);

    // Parse prologue
    parsed.prologue = ParsePrologue(metadata.prologue);

    // Parse each context payload field that is enabled

    // Context Field Change Indicator
    if(vrtGetBit(metadata.indicators, VRT_CNTX_FIELD_CHANGE_BIT)) {
        parsed.indicators.isContextFieldChanged = true;
        curr += VRT_CNTX_FIELD_CHANGE_SIZE;
    }
    // Reference Point Identifier
    if(vrtGetBit(metadata.indicators, VRT_CNTX_REFERENCE_POINT_BIT)) {
        // curr += VRT_CNTX_REFERENCE_POINT_SIZE; // Unused
    }
    // Bandwidth
    if(vrtGetBit(metadata.indicators, VRT_CNTX_BANDWIDTH_BIT)) {
        parsed.indicators.isBandwidth = true;
        parsed.bandwidth = vrtConvertFreqToFloat((VrtFreq)*((int64_t*)curr));
        curr += VRT_CNTX_BANDWIDTH_SIZE;
    }
    // IF Reference Frequency
    if(vrtGetBit(metadata.indicators, VRT_CNTX_IF_FREQ_BIT)) {
        // curr += VRT_CNTX_IF_FREQ_SIZE; // Unused
    }
    // RF Reference Frequency
    if(vrtGetBit(metadata.indicators, VRT_CNTX_RF_FREQ_BIT)) {
        parsed.indicators.isRfFreq = true;
        parsed.rfFreq = vrtConvertFreqToFloat((VrtFreq)*((int64_t*)curr));
        curr += VRT_CNTX_RF_FREQ_SIZE;
    }
    // RF Reference Frequency Offset
    if(vrtGetBit(metadata.indicators, VRT_CNTX_FREQ_OFFSET_BIT)) {
        // curr += VRT_CNTX_FREQ_OFFSET_SIZE; // Unused
    }
    // IF Band Offset
    if(vrtGetBit(metadata.indicators, VRT_CNTX_IF_BAND_OFFSET_BIT)) {
        // curr += VRT_CNTX_IF_BAND_OFFSET_SIZE; // Unused
    }
    // Reference Level
    if(vrtGetBit(metadata.indicators, VRT_CNTX_REFERENCE_LEVEL_BIT)) {
        parsed.indicators.isReflevel = true;
        parsed.reflevel = vrtConvertRefToFloat((VrtRef)*((int16_t*)curr));
        curr += VRT_CNTX_REFERENCE_LEVEL_SIZE;
        reflevel = parsed.reflevel; // Set object's reflevel for calculating amplitude
    }
    // Atten
    if(vrtGetBit(metadata.indicators, VRT_CNTX_GAIN_BIT)) {
        parsed.indicators.isaAtten = true;
        parsed.atten = vrtConvertGainToFloat((VrtGain)*((int16_t*)curr));
        curr += VRT_CNTX_GAIN_SIZE;
    }
    // Over-Range Count
    if(vrtGetBit(metadata.indicators, VRT_CNTX_OVER_RANGE_BIT)) {
        // curr += VRT_CNTX_OVER_RANGE_SIZE; // Unused
    }
    // Sample Rate
    if(vrtGetBit(metadata.indicators, VRT_CNTX_SAMPLE_RATE_BIT)) {
        parsed.indicators.isSampleRate = true;
        parsed.sampleRate = vrtConvertFreqToFloat((VrtFreq)*((int64_t*)curr));
        curr += VRT_CNTX_SAMPLE_RATE_SIZE;
    }
    // Timebase Adjustment
    if(vrtGetBit(metadata.indicators, VRT_CNTX_TIMESTAMP_ADJUST_BIT)) {
        // curr += VRT_CNTX_TIMESTAMP_ADJUST_SIZE; // Unused
    }
    // Timebase Calibration Time
    if(vrtGetBit(metadata.indicators, VRT_CNTX_TIMESTAMP_CAL_BIT)) {
        // curr += VRT_CNTX_TIMESTAMP_CAL_SIZE; // Unused
    }
    // Temperature
    if(vrtGetBit(metadata.indicators, VRT_CNTX_TEMPERATURE_BIT)) {
        parsed.indicators.isTemperature = true;
        parsed.temperature = vrtConvertTempToFloat((VrtTemp)*((int16_t*)curr));
        curr += VRT_CNTX_TEMPERATURE_SIZE;
    }
    // Device Identifier
    if(vrtGetBit(metadata.indicators, VRT_CNTX_DEVICE_ID_BIT)) {
        parsed.indicators.isDevUid = true;
        parsed.indicators.isDevModel = true;
        parsed.devUid = curr[0] & 0x00FFFFFF;
        parsed.devModel = curr[1] & 0x0000FFFF;
        curr += VRT_CNTX_DEVICE_ID_SIZE;
    }
    // State and Event Indicators
    if(vrtGetBit(metadata.indicators, VRT_CNTX_STATE_EVENT_BIT)) {
        // curr += VRT_CNTX_STATE_EVENT_SIZE; // Unused
    }
    // IF Data Packet Payload Format
    if(vrtGetBit(metadata.indicators, VRT_CNTX_IF_PAYLOAD_BIT)) {
        // curr += VRT_CNTX_IF_PAYLOAD_SIZE; // Unused
    }
    // Formatted GPS Geolocation
    if(vrtGetBit(metadata.indicators, VRT_CNTX_FORMATTED_GPS_BIT)) {
        parsed.indicators.isGPS = true;
        VRTFormattedGPS gps;
        memcpy(&gps, curr, sizeof(gps));
        parsed.gps.latitude = vrtConvertFixedToFloat(gps.latitude, 22);
        parsed.gps.longitude = vrtConvertFixedToFloat(gps.longitude, 22);
        parsed.gps.altitude = vrtConvertFixedToFloat(gps.altitude, 6);

        parsed.gps.seconds = gps.timestampInt;
        parsed.gps.picoUpper = gps.timestampFracUpper;
        parsed.gps.picoLower = gps.timestampFracLower;
    }
    curr += VRT_CNTX_FORMATTED_GPS_SIZE; // GPS is left blank when not indicated
    // Formatted INS Geolocation
    if(vrtGetBit(metadata.indicators, VRT_CNTX_FORMATTED_INS_BIT)) {
        // curr += VRT_CNTX_FORMATTED_INS_SIZE; // Unused
    }
    // ECED (Earth-Centered, Earth-Fixed) Ephemeris
    if(vrtGetBit(metadata.indicators, VRT_CNTX_ECEF_BIT)) {
        // curr += VRT_CNTX_ECEF_SIZE; // Unused
    }
    // Relative Ephemeris
    if(vrtGetBit(metadata.indicators, VRT_CNTX_RELATIVE_EPHEMERIS_BIT)) {
        // curr += VRT_CNTX_RELATIVE_EPHEMERIS_SIZE; // Unused
    }
    // Ephemeris Reference Indicator
    if(vrtGetBit(metadata.indicators, VRT_CNTX_EPHEMERIS_REF_BIT)) {
        // curr += VRT_CNTX_EPHEMERIS_REF_SIZE; // Unused
    }
    // GPS ASCII
    if(vrtGetBit(metadata.indicators, VRT_CNTX_GPS_ASCII_BIT)) {
        // curr += VRT_CNTX_GPS_ASCII_SIZE; // Unused
    }

    return curr - base;
}

void VRTParser::Peek(const uint32_t *pkts, SmVRTPacketType *packetType, uint32_t *packetSize)
{
    uint32_t header;
    memcpy(&header, pkts, sizeof(header));
    vrtSwapBytes(&header, sizeof(header));

    uint32_t packetTypeInt = vrtGetPacketType(header);

    switch(vrtGetPacketType(header)) {
    case VRT_DATA_PACKET_CODE:
        *packetType = smVRTDataPacket;
        break;
    case VRT_CONTEXT_PACKET_CODE:
        *packetType = smVRTContextPacket;
        break;
    default:
        *packetType = smVRTInvalidPacket;
        break;
    }

    *packetSize = vrtGetPacketSize(header);
}

VRTUserPktPrologue VRTParser::ParsePrologue(const VRTPktPrologue &prologue)
{
    VRTUserPktPrologue parsed;

    // Header
    parsed.header.packetType = vrtGetPacketType(prologue.header);
    parsed.header.packetCount = vrtGetPacketCount(prologue.header);
    parsed.header.packetSize = vrtGetPacketSize(prologue.header);

    parsed.streamIdent = prologue.streamIdent;
    parsed.seconds = prologue.seconds;
    parsed.picoUpper = prologue.picoUpper;
    parsed.picoLower = prologue.picoLower;

    return parsed;
}

void VRTParser::UnpackSamples(const int16_t* src, float* dst, int len)
{
    double mwReference = pow(10.0, reflevel / 10.0);
    float scaleFactor = sqrt(mwReference);
    for(int i = 0; i < len; i++) {
        dst[i] = ((float)src[i])/pow(2, 15) * scaleFactor;
    }
}
