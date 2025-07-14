#include <cassert>
#include <cstdio>
#include <vector>
#include <string>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates using SCPI to perform a simple S11 measurement. 
// This example does not demonstrate applying a calibration.
// This example retreives both the frequency and formatted data associated with an S11 sweep.
// This example illustrates how to use :INIT:IMM and *OPC? to perform a sweep.

void scpi_vna_s11()
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

    // We assume the software is already running with a device connected

    // Start by presetting the software.
    // This will put the device in s-parameter mode with one channel, one plot, and one S11 trace active.
    viPrintf(inst, (ViString)"*RST\n");

    // Number of points in sweep
    const int N = 401;

    // Configure the stimulus   
    viPrintf(inst, (ViString)":SENSE:FREQ:START 1GHZ; :SENSE:FREQ:STOP 2GHZ\n");
    // IF bandwidth
    viPrintf(inst, (ViString)":SENSE:BAND 10KHZ\n");
    // Sweep points
    viPrintf(inst, (ViString)":SENSE:SWEEP:POINTS %d\n", N);
    
    // Get catalog of traces active, should only be 1, and should be trace 1
    int traceCount = 16;
    std::vector<int> traceList(100);
    viQueryf(inst, (ViString)":CALC:MEAS:CATALOG?\n", (ViString)"%,#d", &traceCount, &traceList[0]);

    assert(traceCount == 1);

    // Set trace format
    viPrintf(inst, (ViString)":CALC:MEAS%d:FORMAT MLINEAR\n", traceList[0]);

    // Trigger a sweep and wait for it
    int opcResult;
    viQueryf(inst, (ViString)":INIT1:IMM; *OPC?\n", (ViString)"%d", &opcResult);
    assert(opcResult == 1);

    // Cache which trace is active. We send it to the following commands for illustrative purposes, despite knowing that it should be trace 1.
    int tr = traceList[0];

    // Query the frequency values for each point
    std::vector<double> freqs(N);
    int sz = N;
    viQueryf(inst, (ViString)":CALC1:MEAS%d:DATA:X?\n", (ViString)"%,#lf", tr, &sz, &freqs[0]);
    assert(sz == N);

    // Query the formatted LinMag data
    std::vector<double> linMag(N);
    sz = N;
    viQueryf(inst, (ViString)":CALC1:MEAS%d:DATA:FDATA?\n", (ViString)"%,#lf", tr, &sz, &linMag[0]);
    assert(sz == N);

    // Do something with data here.

    // Done
    viClose(inst);
}
