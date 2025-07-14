#include <cassert>
#include <cstdio>
#include <vector>
#include <string>
#include <complex>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates using SCPI to read the raw s-parameter data from the instrument
// This example sets up an S11 (1 port only) sweep, and reads the S11 data.
// This example does not include the use of a SOL calibration.

void scpi_vna_read_sparams_1port()
{
    ViSession rm, inst;
    ViStatus rmStatus;

    // Get the VISA resource manager
    rmStatus = viOpenDefaultRM(&rm);
    assert(rmStatus == 0);

    // Open a session to the VNA400 software, VNA400 software must be running at this point
    // Using the default SCPI port for the VNA400 software, which can be changed in the preferences menu.
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

    // At this point only channel 1 is active
    // Let's set the channel to HOLD
    viPrintf(inst, (ViString)":SENSE1:SWEEP:MODE HOLD\n");

    // Now lets configure the stimulus for channel 1
    // Number of points in sweep
    const int N = 401;

    // Configure the stimulus   
    viPrintf(inst, (ViString)":SENSE1:FREQ:START 40MHZ; :SENSE:FREQ:STOP 40GHZ\n");
    // IF bandwidth
    viPrintf(inst, (ViString)":SENSE1:BAND 10KHZ\n");
    // Sweep points
    viPrintf(inst, (ViString)":SENSE1:SWEEP:POINTS %d\n", N);
    // Averaging
    viPrintf(inst, (ViString)":SENSE1:AVERAGE:STATE OFF\n");

    // Get catalog of traces active, should only be 1, and should be trace 1.
    // We know this because we preset the instrument and that's the default.
    int traceCount = 16;
    std::vector<int> traceList(100);
    viQueryf(inst, (ViString)":CALC:MEAS:CATALOG?\n", (ViString)"%,#d", &traceCount, &traceList[0]);
    assert(traceCount == 1);

    // Set trace parameter
    viPrintf(inst, (ViString)":CALC1:MEAS1:PARAMETER S11\n");
    // Set trace format
    viPrintf(inst, (ViString)":CALC1:MEAS1:FORMAT MLOG\n");

    // Trigger a sweep and wait for it
    int opcResult;
    viQueryf(inst, (ViString)":INIT1:IMM; *OPC?\n", (ViString)"%d", &opcResult);
    assert(opcResult == 1);

    // Query the frequency values for each point
    // These do not change unless the stimulus frequency range is changed.
    std::vector<double> freqs(N);
    int sz = N;
    viQueryf(inst, (ViString)":CALC1:MEAS1:DATA:X?\n", (ViString)"%,#lf", &sz, &freqs[0]);
    assert(sz == N);

    // Query the S11 data, returned as complex data, 2 values (I/Q) for each freq point.
    std::vector<std::complex<double>> s11(N);
    // Tell VISA to look for N*2 points, since complex.
    sz = N*2;
    viQueryf(inst, (ViString)":CALC1:MEAS1:DATA:SDATA?\n", (ViString)"%,#lf", &sz, &s11[0]);
    // Verify we have received all the data.
    assert(sz == N*2);

    // Do something with data here.

    // Done
    viClose(inst);
}
