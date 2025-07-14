#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Configure a limit line to match the current sweep span of the device.
// 3. Perform a single sweep
// 4. Test whether the limit line test passed/failed

// No input signal required

void scpi_configure_lline()
{
    ViSession rm, inst;
    ViStatus rmStatus;
    int opc;

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

    // Set the measurement mode to Digital Modulation Analysis
    viPrintf(inst, "INSTRUMENT:SELECT SA\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");
    viQueryf(inst, "*OPC?\n", "%d", &opc);

    // Get the current frequency range of the device, we will create a limit line with this
    //  frequency range as well.
    double startFreq, stopFreq;
    viQueryf(inst, ":FREQ:START?; :FREQ:STOP?\n", "%lf;%lf", &startFreq, &stopFreq);

    // Configure, upper limit with no offset
    viPrintf(inst, ":CALC:LLINE1:STATE ON\n");
    viPrintf(inst, ":CALC:LLINE1:TRACE 1\n");
    viPrintf(inst, ":CALC:LLINE1:TYPE UPPER\n");
    viPrintf(inst, ":CALC:LLINE1:OFFSET:Y 0.0\n");
    viPrintf(inst, ":CALC:LLINE1:STATE ON\n");

    // Create limit line at -50 dBm across full freq range
    double lline[4];
    lline[0] = startFreq;
    lline[1] = -50;
    lline[2] = stopFreq;
    lline[3] = -50;

    // Load limit line
    viPrintf(inst, ":CALC:LLINE1:DATA %,4lf\n", &lline[0]);

    // Test
    int fail = 0;
    viQueryf(inst, ":INIT; *OPC; :CALC:LLINE1:FAIL?\n", "%d", &fail);

    // No longer need limit lines, clear them
    viPrintf(inst, ":CALC:LLINE1:CLEAR\n");

    // Done
    viClose(inst);
}