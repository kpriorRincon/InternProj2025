#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Put the device into non-continuous mode 
// 4. Perform and wait for a sweep to complete

void scpi_network_analyzer_sweep()
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

    // Set the measurement mode to sweep
    viPrintf(inst, "INSTRUMENT:SELECT NA\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");

    // Configure a 20MHz span sweep at 1GHz
    // Center/span
    viPrintf(inst, "SENS:FREQ:START 100MHZ; STOP 4GHZ\n");
    // Fast sweep with 100 points
    viPrintf(inst, ":NA:SWEEP:POINTS 100; TYPE PASSIVE; HRANGE ON\n");
    // Configure view
    viPrintf(inst, ":NA:VIEW:SCALE LOG; RLEVEL 10; DIV 10\n");

    // Configure the trace. Ensures trace 1 is active and enabled for clear-and-write.
    // These commands are not required to be sent everytime, this is for illustrative purposes only. 
    viPrintf(inst, "TRAC:SEL 1\n"); // Select trace 1
    viPrintf(inst, "TRAC:TYPE WRITE\n"); // Set clear and write mode
    viPrintf(inst, "TRAC:UPD ON\n"); // Set update state to on
    viPrintf(inst, "TRAC:DISP ON\n"); // Set un-hidden

    // Trigger a sweep, and wait for it to complete
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

    // Tell the software to perform a store through on the next sweep
    viPrintf(inst, ":CORR:NA:STORE:THRU\n");

    // The sweep after the store through calibrates the sweep path.
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

    // At this point you would insert the DUT and subsequent sweeps would be calibrated
    int calibrated = 0;
    viQueryf(inst, "CORR:NA:STORE:THRU:ACTIVE?\n", "%d", &calibrated);

    // Do another sweep
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

    // Done
    viClose(inst);
}