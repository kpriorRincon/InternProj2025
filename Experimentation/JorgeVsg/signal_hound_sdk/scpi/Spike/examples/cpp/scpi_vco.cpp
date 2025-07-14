#include <cassert>
#include <cstdio>
#include <vector>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike
// 2. Configuring Spike for VCO characterization measurements
// 3. Performing a full VCO characterization sweep measurement
// 4. Querying the trace data of the resulting measurements

// Equipment needed is a BB, SP, or SM series spectrum analyzer, a PN400, and a VCO to test

void scpi_vco()
{
    ViSession rm, inst;
    ViStatus rmStatus;

    // Get the VISA resource manager
    rmStatus = viOpenDefaultRM(&rm);
    assert(rmStatus == 0);

    // Open a session to the Spike software, Spike must be running at this point
    ViStatus instStatus = viOpen(rm, "TCPIP::localhost::5025::SOCKET", VI_NULL, VI_NULL, &inst);
    assert(instStatus == 0);

    // For SOCKET programming, we want to tell VISA to use a terminating character 
    //   to end a read operation. In this case we want the newline character to end a 
    //   read operation. The termchar is typically set to newline by default. Here we
    //   set it for illustrative purposes.
    viSetAttribute(inst, VI_ATTR_TERMCHAR_EN, VI_TRUE);
    viSetAttribute(inst, VI_ATTR_TERMCHAR, '\n');

    // Set the measurement mode to VCO Characterization
    viPrintf(inst, "INSTRUMENT:SELECT VCO\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");

    const double START = 1.0;
    const double STOP = 17.0;
    const int POINTS = 100;

    // Configure sweep    
    viPrintf(inst, "SENS:VCO:SWEEP:START %.2f; :VCO:SWEEP:STOP %.2f; :VCO:SWEEP:POINTS %d\n", START, STOP, POINTS);
    viPrintf(inst, "SENS:VCO:SWEEP:RLEV -10; :VCO:SWEEP:FCO:RES 10KHZ; :VCO:SWEEP:DEL 1\n");

    // Configure source
    viPrintf(inst, "SENS:VCO:SOUR:VOLT:VTUN:LIM:LOW -1.0; :VCO:SOUR:VOLT:VTUN:LIM:HIGH 28.0\n");
    viPrintf(inst, "SENS:VCO:SOUR:VOLT:VSUP:LIM:LOW 0.5; :VCO:SOUR:VOLT:VSUP:LIM:HIGH 15.25\n");
    viPrintf(inst, "SENS:VCO:SOURCE:VOLT:FIXED 5.0\n");
    viPrintf(inst, "SENS:VCO:SOURCE:VOLT ON\n");

    // Perform a measurement
    int opc;
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);
    assert(opc == 1);

    // Retrieve the measurement results

    // Preallocate arrays for the measurements
    std::vector<float> frequency(POINTS);
    std::vector<float> sensitivity(POINTS);
    std::vector<float> power(POINTS);
    std::vector<float> current(POINTS);
    const int HARMONIC_COUNT = 6;
    std::vector<std::vector<float>> harmonics(HARMONIC_COUNT, std::vector<float>(POINTS));

    int length = POINTS;

    // Request data
    // Measurement data is returned as comma separated values
    viQueryf(inst, "FETCh:VCO:FREQ?\n", "%,#f", &length, frequency.data());
    viQueryf(inst, "FETCh:VCO:SENS?\n", "%,#f", &length, sensitivity.data());
    viQueryf(inst, "FETCh:VCO:POW?\n", "%,#f", &length, power.data());
    viQueryf(inst, "FETCh:VCO:CURR?\n", "%,#f", &length, current.data());
    for(int harm = 0; harm < HARMONIC_COUNT; harm++) {
        viQueryf(inst, "FETCh:VCO:HARM? %d\n", "%,#f", harm + 1, &length, harmonics[harm].data());
    }    

    // Print out meas information
    const double vStep = (STOP - START) / POINTS;
    for(int i = 0; i < POINTS; i++) {
        printf("V Tune: %.2f V\n\tFreq: %.4f MHz\n\tSensitivity: %.4f Hz\n\tPower: %.4f dBm\n\tCurrent: %.4f mA\n\t", 
            START + i * vStep, frequency[i] / 1e6, sensitivity[i], power[i], current[i]);
        for(int harm = 0; harm < HARMONIC_COUNT; harm++) {
            printf("Harmonic %d: %.4f dBm\n\t", harm + 1, harmonics[harm][i]);
        }
        printf("\n");
    }

    // Done
    viClose(inst);
}