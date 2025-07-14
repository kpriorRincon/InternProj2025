// Copyright Signal Hound 2018

// This file demonstrates how to parse the SHR sweep recording files generated
//   by the Spike software.

// File header version info.
// Version 1: Initial release.
// Version 2: Added ADC overflow bit into the sweep header reserved section.

#pragma once

#include <cstdint>
#include <string>

// The basic format of the SHR file is shown below
//
// SHRFileHeader
// SHRSweepHeader[1]
// SweepData[1]
// SHRSweepHeader[2]
// SweepData[2]
// ...
// SHRSweepHeader[N]
// SweepData[N]
//
// The SHRFileHeader contains the following information
//  - The device configuration at the time of capture
//  - The decimation settings at the time of capture.
//  - Any channelizer effects at the time of capture
//
// The SHRFileHeader is simply memcpy'ed to and from the file starting at a byte 
//   offset of 0 from the beginning of the file.
// The SHRSweepHeader is simply memcpy'ed to/from the file.
// The sweep data is an array of SHRFileHeader::sweepLength contiguous 32-bit floating
//   point values that are memcpy'ed to the file.
//
// Please read the following bullet points which describe key features of the file format
// - The file signature is the first two bytes of the file, and must match the
//   SHRFileSignature
// - The file is stored as little endian
// - 

const uint16_t SHRFileSignature = 0xAA10;
const uint16_t SHRFileVersion = 0x2;

enum SHRScale {
    SHRScaleDBM = 0,
    SHRScaleMV = 1
};

enum SHRWindow {
    SHRWindowNuttall = 0,
    SHRWindowFlattop = 1,
    SHRWindowGaussian = 2
};

enum SHRDecimationType {
    SHRDecimationTypeTime,
    SHRDecimationTypeCount
};

enum SHRDecimationDetector {
    SHRDecimationDetectorAvg = 0,
    SHRDecimationDetectorMax = 1
};

enum SHRChannelizerOutputUnits {
    SHRChannelizerOutputUnitsDBM = 0,
    SHRChannelizerOutputUnitsDBMHz = 1
};

enum SHRVideoDetector {
    SHRVideoDetectorMinMax = 0,
    SHRVideoDetectorAvg = 1
};

enum SHRVideoUnits {
    SHRVideoUnitsLog = 0,
    SHRVideoUnitsVoltage = 1,
    SHRVideoUnitsPower = 2,
    SHRVideoUnitsSample = 3
};

#pragma pack(push,1)
struct SHRFileHeader {
    uint16_t signature;
    uint16_t version;
    uint32_t reserved1;

    // Byte offset where the sweep data starts
    uint64_t dataOffset;

    // Sweep parameters
    uint32_t sweepCount; // Number of sweeps in file
    uint32_t sweepLength; // Size of each sweep
    double firstBinFreqHz;
    double binSizeHz;

    // 127 total characters plus NULL
    uint16_t title[128];

    // Device configuration at time of capture
    double centerFreqHz;
    double spanHz;
    double rbwHz;
    double vbwHz;
    float refLevel; // dBm/mV depending on scale
    uint32_t refScale; // SHRScale
    float div;
    uint32_t window; // SHRWindow
    int32_t attenuation;
    int32_t gain;
    int32_t detector; // SHRVideoDetector values
    int32_t processingUnits; // SHRVideoUnits values
    double windowBandwidth; // in bins

    // Time averaging configuration
    int32_t decimationType; // SHRDecimationType
    int32_t decimationDetector; // SHRDecimationDetector
    int32_t decimationCount;
    int32_t decimationTimeMs;

    // Frequency averaging configuration
    int32_t channelizeEnabled; // 1 for enabled, 0 for disabled
    int32_t channelOutputUnits; //
    double channelCenterHz;
    double channelWidthHz;

    uint32_t reserved2[16];
};
#pragma pack(pop)

#pragma pack(push,1)
struct SHRSweepHeader {
    uint64_t timestamp; // milliseconds since epoch
    double latitude;
    double longitude;
    double altitude; // meters
    uint8_t adcOverflow;
    uint8_t reserved[15];
};
#pragma pack(pop)

// These functions and data types are an example of parsing the SHR files
//   using the fopen/fseek/fread/fclose functions.
// These are by no means the only approach to doing so.

struct SHRParseState {
    SHRParseState() { f = nullptr; }

    FILE *f;
    std::string fileName;
    SHRFileHeader header;
};

// Open an SHR file, return true if success
// If success, state will contain a valid parse state which can be used for the
//   SHRGetSweep function.
bool SHROpenFile(const std::string &fileName, SHRParseState &state);
// state = valid parser state struct
void SHRCloseFile(SHRParseState &state);

// state = valid parser state struct
// n = the sweep you want to retrieve from [0, state.header.sweepCount-1]
// sweep = preallocated array of floats, will store the sweep if function returns 
//   successfully. Should be state.header.sweepLength number of floats in size
// sweepInfo = timestamp and position of sweep if successful
bool SHRGetSweep(SHRParseState &state, int n, float *sweep, SHRSweepHeader &sweepInfo);