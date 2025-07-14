#include <cassert>
#include <cstdio>
#include <vector>
#include <string>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates using SCPI to setup a 2 port sweep and retrieve all 4 S-parameters
// This example illustrates how to configure a plot and multiple traces.
// This example does not illustrate how to load a calibration.
// This example illustrates how to use :INIT:IMM and *OPC? to perform a sweep.
// This example illustrates how to retreive the s-parameter data from a trace.

void scpi_vna_2port()
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

    // Assume the software is already running with a device connected

    // Start by presetting the software.
        // This will put the device in s-parameter mode with one channel, one plot, and one S11 trace active.
    viPrintf(inst, (ViString)"*RST\n");

    // Set to hold to start
    viPrintf(inst, (ViString)":SENSE1:SWEEP:MODE HOLD\n");

    // Delete all traces
    viPrintf(inst, (ViString)":CALC:MEAS:DELETE:ALL\n");

    // Query all active plots and delete them all. This is as an exercise only since we know that there should be exactly
    //   one plot after issuing the reset command above.
    std::vector<int> displayCatalog(16);
    int displayCatalogSz = displayCatalog.size();
    viQueryf(inst, (ViString)":DISPLAY:CATALOG?\n", (ViString)"%,#d", &displayCatalogSz, &displayCatalog[0]);
    // Loop through all returned plot numbers and close them
    for(int i = 0; i < displayCatalogSz; i++) {
        viPrintf(inst, (ViString)":DISPLAY:WINDOW%d:STATE OFF\n", displayCatalog[i]);
    }

    // Channel 1 is active after a reset, we don't do any setup with the channels, just use channel 1.

    // Add/enable one plot
    viPrintf(inst, (ViString)":DISPLAY:WINDOW1:STATE ON\n");

    const char *paramNames[4] = {
        "S11", "S21", "S12", "S22"
    };

    // Add 4 traces
    for(int i = 0; i < 4; i++) {
        // Enable trace
        viPrintf(inst, (ViString)":CALC1:MEAS%d:DEFINE\n", i+1);
        // Set to display 1
        viPrintf(inst, (ViString)":CALC1:MEAS%d:DISPLAY 1\n", i+1);
        // Set s-parameter
        viPrintf(inst, (ViString)":CALC1:MEAS%d:PARAMETER %s\n", i+1, paramNames[i]);
        // Set format (LogMag)
        viPrintf(inst, (ViString)":CALC1:MEAS%d:FORMAT MLOG\n", i+1);
    }

    // Check for errors
    int errorCount = 0;
    viQueryf(inst, (ViString)"SYSTEM:ERROR:COUNT?\n", (ViString)"%d", &errorCount);
    assert(errorCount == 0);

    // Configure the sweep
    // Number of points in sweep
    const int N = 201;

    // Configure the stimulus
    // Center/span    
    viPrintf(inst, (ViString)":SENSE1:FREQ:START 10GHZ; :SENSE:FREQ:STOP 30GHZ\n");
    // IF bandwidth
    viPrintf(inst, (ViString)":SENSE1:BAND 100KHZ\n");
    // Sweep points
    viPrintf(inst, (ViString)":SENSE1:SWEEP:POINTS %d\n", N);
    // Averaging
    viPrintf(inst, (ViString)":SENSE1:AVERAGE:STATE OFF\n");
    // Output power
    viPrintf(inst, (ViString)":SOURce1:POWER:MODE HIGH\n");

    // Trigger a sweep and wait for it
    int opcResult;
    viQueryf(inst, (ViString)":INIT1:IMM; *OPC?\n", (ViString)"%d", &opcResult);
    assert(opcResult == 1);

    // Query the frequency values for each point
    // Only need to do this for one trace as they should all be mapped to the same frequencies
    std::vector<double> freqs(N);
    int sz = N;
    viQueryf(inst, (ViString)":CALC1:MEAS1:DATA:X?\n", (ViString)"%,#lf", &sz, &freqs[0]);
    assert(sz == N);

    // Query the s-parameter data
    std::vector<double> sparams[4];
    for(int i = 0; i < 4; i++) {
        // Allocate for N complex values, interleaved real/imag
        sz = N * 2;
        sparams[i].resize(sz);
        viQueryf(inst, (ViString)":CALC1:MEAS%d:DATA:SDATA?\n", (ViString)"%,#lf", i + 1, &sz, &sparams[i][0]);
        assert(sz == N * 2);
    }

    // Done
    viClose(inst);
}
