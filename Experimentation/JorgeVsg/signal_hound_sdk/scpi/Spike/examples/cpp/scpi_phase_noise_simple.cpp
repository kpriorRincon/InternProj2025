#include <cassert>
#include <cstdio>
#include <vector>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike
// 2. Configuring a phase noise measurement
// 3. Performing several sweeps to generate an average phase noise trace
// 4. Using the marker to create a decade table

// Intended input signal is a 1GHz CW above the amplitude threshold set in the software

void scpi_phase_noise_simple()
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

    // Select phase noise measurement mode
    viPrintf(inst, "INSTRUMENT:SELECT PN\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");

    // Signal search configuration
    viPrintf(inst, "PN:CARRIER:SEARCH 1\n");
    // Set to small/large values to force clamp to range of instrument
    viPrintf(inst, "PN:CARRIER:SEARCH:START 1Hz\n");
    viPrintf(inst, "PN:CARRIER:SEARCH:STOP 99GHz\n");
    viPrintf(inst, "PN:CARRIER:THR:MIN -40\n");

    // Measure phase noise only
    viPrintf(inst, "PN:TYPE PN\n");
    // 100Hz to 10MHz offsets
    viPrintf(inst, "PN:FREQ:OFFS:STAR 100Hz; STOP 10MHz\n");

    // Configure plot
    viPrintf(inst, "PN:VIEW:RLEV -50; PDIV 10\n");

    int averageCount = 5;

    // Phase noise measurements can take awhile, rather than spin a loop waiting for 
    //  a non-timeout, lets just increase our timeout value for the measurements.
    // The sweep time is measured in Spike first to determine a reasonable timeout val.
    viSetAttribute(inst, VI_ATTR_TMO_VALUE, 10e3);

    // Setup the traces
    viPrintf(inst, "TRAC:PN:SEL 1; TYPE NORMAL\n");
    viPrintf(inst, "TRAC:PN:SEL 2; TYPE AVERAGE; AVER:COUNT 5\n");

    // Do a single sweep and set it as the reference
    // For instructive purposes only
    int opc;
    viQueryf(inst, "INIT; *OPC?\n", "%d", &opc);
    viPrintf(inst, "TRAC:PN:SEL 1; TO 3\n"); // Store the reference
    viPrintf(inst, "TRAC:PN:SEL 2; CLEAR\n"); // Clear the average trace

    // Now do the sweeps, wait for each one to complete
    for(int i = 0; i < averageCount; i++) {
        int opc;
        viQueryf(inst, "INIT; *OPC?\n", "%d", &opc);
    }

    // Get decade marker tables
    const int tblSize = 6;
    double offsets[tblSize] = {100, 1e3, 10e3, 100e3, 1e6, 10e6};
    double table[tblSize];

    viPrintf(inst, "CALC:PN:MARKER:SELECT 1");
    viPrintf(inst, "CALC:PN:MARK ON\n");

    for(int m = 0; m < 3; m++) {
        viPrintf(inst, "CALCULATE:PNOISE:MARKER:TRACE %d\n", m+1);
        for(int i = 0; i < tblSize; i++) {
            viPrintf(inst, "CALC:PN:MARK:X %fHz\n", offsets[i]);
            viQueryf(inst, "CALC:PN:MARK:Y?\n", "%lf", table + i);
        }

        // Print it off to the console
        printf("Decade table for trace %d\n", m);
        for(int i = 0; i < tblSize; i++) {
            printf("%g Offset: %f dBc\n", offsets[i], table[i]);
        }
    }

    viClose(inst);
}