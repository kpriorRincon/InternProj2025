#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Configure a path loss table with several points
// 3. Perform a single sweep
// 4. Perform a peak search with path loss applied

// No input signal required

void scpi_path_loss_table()
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

    // Set the measurement mode to Swept Analysis
    viPrintf(inst, "INSTRUMENT:SELECT SA\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");
    viQueryf(inst, "*OPC?\n", "%d", &opc);

	// Configure path loss table 1
	viPrintf(inst, "SENS:CORR:PATH1:CLEAR\n"); // Clear table
	viPrintf(inst, "SENS:CORR:PATH1:DESC SCPI Demo\n"); // Set table name	
	viPrintf(inst, "SENS:CORR:PATH1:STATE ON\n"); // Activate table
	
	// Load table data:
	//
	// Freq (GHz) | Offset (dB)
	// -----------|------------
	//          1 | 10
	//			2 | 20
	//			3 | 30
	double table[6] = { 1e9, 10, 2e9, 20, 3e9, 30 };
	viPrintf(inst, "SENS:CORR:PATH1:DATA %,6lf\n", table);

	// Get a sweep with table applied
	viPrintf(inst, ":INIT\n");
	viQueryf(inst, "*OPC?\n", "%d", &opc);

	// Find peak
	double peak;
	viQueryf(inst, "CALC:MARK:MAX; Y?\n", "%lf", &peak);

    // Reset table
	viPrintf(inst, "SENS:CORR:PATH1:CLEAR\n");
	viPrintf(inst, "SENS:CORR:PATH1:DESC Antenna Factor\n");
	viPrintf(inst, "SENS:CORR:PATH1:STATE OFF\n");

    // Done
    viClose(inst);
}
