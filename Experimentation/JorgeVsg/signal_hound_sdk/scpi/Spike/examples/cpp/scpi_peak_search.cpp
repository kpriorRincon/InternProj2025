#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example is an extension of the scpi_simple_sweep

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Put the device into non-continuous mode 
// 3. Configure the marker and trace
// 4. Perform and wait for a sweep to complete
// 5. Peak search and output peak information

// The intended input tone for this example is a CW at 1GHz below -20dBm amplitude

void scpi_peak_search()
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
    viPrintf(inst, "SENS:SWE:DET:FUNC MINMAX; UNIT POWER\n");

    // Configure the trace
    // These commands are not required to be sent everytime, this is for illustrative
    //   purposes only. Ensures trace 1 is active and enabled for clear-and-write.
    viPrintf(inst, "TRAC:SEL 1\n"); // Select trace 1
    viPrintf(inst, "TRAC:TYPE WRITE\n"); // Set clear and write mode
    viPrintf(inst, "TRAC:UPD ON\n"); // Set update state to on
    viPrintf(inst, "TRAC:DISP ON\n"); // Set un-hidden

    // Configure the marker
    // Below is a blown out list of commands that could be sent to ensure the marker is configured
    //   properly. Not all of these commands are required to be sent everytime. The commands
    //   below are used for illustrative purposes only.
    viPrintf(inst, "CALC:MARK:SEL 1\n"); // Select marker 1
    viPrintf(inst, "CALC:MARK:TRACE 1\n"); // Put marker on trace 1
    viPrintf(inst, "CALC:MARK:MODE POS\n"); // Set marker to position mode
    viPrintf(inst, "CALC:MARK:DELTA OFF\n"); // Disable delta marker
    viPrintf(inst, "CALC:MARK:PKTR OFF\n"); // Disable peak tracking if enabled

    // Trigger a sweep, and wait for it to complete
    int opc;
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

    // Perform a peak search, and query the frequency and amplitude
    // If the marker is not currently enabled, then it is enabled on peak search
    viPrintf(inst, "CALC:MARK:MAX; X?; Y?\n");
    double x, y;
    viScanf(inst, "%lf;%lf", &x, &y);

    printf("Marker Peak: %.6f MHz, Ampl: %.2f dBm\n", x/1.0e6, y);

    // Done
    viClose(inst);
}