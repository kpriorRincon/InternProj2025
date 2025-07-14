#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to the Spike software
// 2. Configuring a sweep
// 3. Configuring the occupied bandwidth measurement
// 4. Performing a sweep
// 5. Performing the occupied bandwidth measurement

// The intended input tone for this example is a modulated signal at 1GHz below -20dBm amplitude

void scpi_occupied_bandwidth()
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

    // Get the VISA resource manager
    rmStatus = viOpenDefaultRM(&rm);
    assert(rmStatus == 0);

    // Set the measurement mode to sweep
    viPrintf(inst, "INSTRUMENT:SELECT SA\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");

    // Configure a 20MHz span sweep at 1GHz
    // Set the RBW/VBW to auto
    viPrintf(inst, "SENS:BAND:RES:AUTO ON; :BAND:VID:AUTO ON; :BAND:SHAPE FLATTOP\n");
    // Center/span
    viPrintf(inst, "SENS:FREQ:SPAN 20MHZ; CENT 1GHZ\n");
    // Reference level/Div
    viPrintf(inst, "SENS:POW:RF:RLEV -20DBM; PDIV 10\n");
    // Peak detector
    viPrintf(inst, "SENS:SWE:DET:FUNC AVERAGE; UNIT POWER\n");

    // Set up occupied bandwidth
    // This has to be setup before we sweep
    viPrintf(inst, "SENSE:OBWIDTH:STATE ON; TRACE 1; PERCENT 95\n");

    // Do a sweep
    int opc;
    viQueryf(inst, "INIT:IMM; *OPC?\n", "%d", &opc);
    assert(opc == 1);

    // Get the center channel power as dBm
    // Get both adjacent channels as dB difference from center channel
    double obWidth, obCenter, obPower;
    viQueryf(inst, "SENSE:OBWIDTH:OBWIDTH?; CENTER?; POWER?\n",
        "%lf;%lf;%lf", &obWidth, &obCenter, &obPower);

    printf("OB Width %f MHz\n", obWidth / 1.0e6);
    printf("OB Center %f MHz\n", obCenter / 1.0e6);
    printf("OB Power %f dBm\n", obPower);

    // Done
    viClose(inst);
}