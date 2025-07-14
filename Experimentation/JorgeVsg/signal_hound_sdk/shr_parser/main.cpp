#include <cstdio>
#include <string>
#include <vector>

#include "shr_parse.h"

int main(int argc, char **argv)
{
    std::string fileName = "example.shr";

    SHRParseState state;
    if(!SHROpenFile(fileName, state)) {
        printf("Unable to open file\n");
        return -1;
    }

    // Print basic sweep info
    printf("File Name: %s\n", fileName.c_str());
    printf("Sweep Count: %d\n", state.header.sweepCount);
    printf("Sweep Size: %d\n", state.header.sweepLength);
    printf("Sweep Start Freq: %f\n", state.header.firstBinFreqHz);
    printf("Sweep Bin Size: %f\n", state.header.binSizeHz);
    printf("Sweep Freq Range: %f MHz to %f MHz\n", 
        (state.header.centerFreqHz - state.header.spanHz / 2.0) * 1.0e-6,
        (state.header.centerFreqHz + state.header.spanHz / 2.0) * 1.0e-6);
    printf("RBW: %f kHz\n", state.header.rbwHz * 1.0e-3);
    printf("VBW: %f kHz\n", state.header.vbwHz * 1.0e-3);
    printf("Reference Level: %f %s\n", state.header.refLevel, 
        (state.header.refScale == SHRScaleDBM) ? "dBm" : "mV");

    // Print decimation info
    if(state.header.decimationType == SHRDecimationTypeCount) {
        if(state.header.decimationDetector == SHRDecimationDetectorAvg) {
            printf("Averaged %d trace(s) per output trace\n", state.header.decimationCount);
        } else {
            printf("Max held %d trace(s) per output trace\n", state.header.decimationCount);
        }
    } else {
        if(state.header.decimationDetector == SHRDecimationDetectorAvg) {
            printf("Averaged trace(s) for %.2f seconds per output trace\n", state.header.decimationCount);
        } else {
            printf("Max held trace(s) for %.2f seconds per output trace\n", state.header.decimationCount);
        }
    }
    printf("%s\n", state.header.channelizeEnabled ? "Was channelized" : "Was not channelized");

    // Get sweeps
    // Setup up sweep buffer
    std::vector<float> sweep(state.header.sweepLength);

    // Loop through all sweeps
    for(int i = 0; i < state.header.sweepCount; i++) {
        SHRSweepHeader sweepInfo;
        SHRGetSweep(state, i, &sweep[0], sweepInfo);

        // Get peak for each sweep, both the frequency and amplitude
        double peakIndex = 0;
        for(int j = 1; j < sweep.size(); j++) {
            if(sweep[j] > sweep[peakIndex]) {
                peakIndex = j;
            }
        }

        double peakFreq = state.header.firstBinFreqHz + peakIndex * state.header.binSizeHz;
        peakFreq /= 1.0e6;
        printf("Sweep %d: Peak Freq %.6f MHz, Peak Ampl %.2f %s\n",
            i, 
            peakFreq, 
            sweep[peakIndex],
            (state.header.refScale == SHRScaleDBM) ? "dBm" : "mV");
    }

    SHRCloseFile(state);

    return 0;
}