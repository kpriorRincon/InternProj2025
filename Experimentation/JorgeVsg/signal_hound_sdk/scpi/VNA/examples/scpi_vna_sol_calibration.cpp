#include <cassert>
#include <cstdio>
#include <vector>
#include <string>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates using SCPI to perform a SOL calibration on port 1.
// This example illustrates
// - Configuring the calibration
// - Measuring the 3 standards
// - Finishing the calibration and saving it.
// This example does not illustrate making measurements once the calibration has been performed.

void scpi_vna_sol_calibration()
{
    ViSession rm, inst;
    ViStatus rmStatus;

    // Get the VISA resource manager
    rmStatus = viOpenDefaultRM(&rm);
    assert(rmStatus == 0);

    // Open a session to the VNA400 software, VNA400 software must be running at this point
    ViStatus instStatus = viOpen(rm, (ViRsrc)"TCPIP::localhost::5026::SOCKET", VI_NULL, VI_NULL, &inst);
    assert(instStatus == 0);

    // For SOCKET programming, we want to tell VISA to use a terminating character 
    //   to end a read operation. In this case we want the newline character to end a 
    //   read operation. The termchar is typically set to newline by default. Here we
    //   set it for illustrative purposes.
    viSetAttribute(inst, VI_ATTR_TERMCHAR_EN, VI_TRUE);
    viSetAttribute(inst, VI_ATTR_TERMCHAR, '\n');

    // Assume the software is already running with a device connected.

    // Start by presetting the software.
    // This will put the device in s-parameter mode with one channel, one plot, and one S11 trace active.
    viPrintf(inst, (ViString)"*RST\n");

    // For this example we will use the default stimulus as our measurement stimulus. 
    // We will also assume channel 1 for all measurements.
    // If you want a different stimulus/setup, you should configure that now. 

    // Specify which cal kit to use to perform the calibration
    viPrintf(inst, (ViString)":SENSE1:CORRECTION:COLLECT:CKIT:SELECT 85056D\n");

    // Start the calibration, specify a SOL calibration on port 1
    viPrintf(inst, (ViString)":SENSE1:CORRECTION:COLLECT:METHOD SOL,1\n");

    int success = false;

    // Now we need to measure the standards, prompt the user to connect each standard prior to measuring it.

    //prompt("Please connect the short to port 1\n");
    // Measure short on port 1 with cal kit std 1
    viQueryf(inst, (ViString)":SENSE1:CORRECTION:COLLECT:ACQUIRE:SHORT? 1,1\n", (ViString)"%d", &success);
    assert(success);

    //prompt("Please connect the open to port 1\n");
    // Measure open on port 1 with cal kit std 2
    viQueryf(inst, (ViString)":SENSE1:CORRECTION:COLLECT:ACQUIRE:OPEN? 1,2\n", (ViString)"%d", &success);
    assert(success);

    //prompt("Please connect the load to port 1\n");
    // Measure open on load 1 with cal kit std 3
    viQueryf(inst, (ViString)":SENSE1:CORRECTION:COLLECT:ACQUIRE:LOAD? 1,3\n", (ViString)"%d", &success);
    assert(success);

    // All measurements performed, save and apply the calibration
    viPrintf(inst, (ViString)":SENSE1:CORRECTION:COLLECT:SAVE \"C:/Users/AJ/Documents/SignalHound/vna/cals/scpi_cal.vnacal\"\n");

    // Check for any errors
    
    // Query the number of errors present
    int errorCount = 0;
    viQueryf(inst, (ViString)":SYSTEM:ERROR:COUNT?\n", (ViString)"%d", &errorCount);
    printf("Error count %d\n", errorCount);

    // For each error, query the error string, print it out in the console
    char errBuf[256];
    for(int i = 0; i < errorCount; i++) {
        viQueryf(inst, (ViString)":SYST:ERR:NEXT?\n", (ViString)"%[^\n]", errBuf);
        printf("Error: %s\n", errBuf);
    }

    // Done
    viClose(inst);
}
