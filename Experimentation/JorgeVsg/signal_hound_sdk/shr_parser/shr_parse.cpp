#include "shr_parse.h"

#pragma warning(disable:4996)

bool SHROpenFile(const std::string &fileName, SHRParseState &state)
{
    if(state.f) { return false; } // Already open

    state.f = fopen(fileName.c_str(), "rb");
    if(!state.f) { return false; } // Unable to open file

    size_t bytesRead = fread(&state.header, 1, sizeof(SHRFileHeader), state.f);
    if(bytesRead != sizeof(SHRFileHeader)) { 
        SHRCloseFile(state);
        return false; // Couldn't read header
    }

    if(state.header.signature != SHRFileSignature 
        || state.header.version > SHRFileVersion)
    {
        SHRCloseFile(state); // Invalid file
        return false;
    }

    return true;
}

void SHRCloseFile(SHRParseState &state)
{
    if(state.f) {
        fclose(state.f);
        state.f = nullptr;
    }
}

bool SHRGetSweep(SHRParseState &state, int n, float *sweep, SHRSweepHeader &sweepInfo)
{
    if(!state.f) { return false; } // Invalid file handle

    size_t sweepSizeInBytes = sizeof(float) * state.header.sweepLength 
        + sizeof(SHRSweepHeader);

    // Move to beginning of sweep
    // Currently the state.header.dataOffset is the same size as the file header itself.
    // This may not always be the case. Use the data offset.
    fseek(state.f, state.header.dataOffset + sweepSizeInBytes * n, SEEK_SET);

    size_t headerBytesRead = fread(&sweepInfo, 1, sizeof(SHRSweepHeader), state.f);
    size_t sweepBytesRead = fread(sweep, 1, sizeof(float) * state.header.sweepLength, state.f);

    return (headerBytesRead == sizeof(SHRSweepHeader)) && (sweepBytesRead == sweepSizeInBytes);
}