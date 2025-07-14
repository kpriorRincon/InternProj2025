#include <cassert>
#include <cstdio>
#include <vector>
#include <string>
#include <complex>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates using SCPI to configure a "CW like" sweep and retrieve
// the formatted S21 sweep data. "CW like" is specified since the software does not
// currently have a true CW mode, setting start = stop causes the software to
// measure at the same frequency for each point in the sweep.

void scpi_vna_cw_sweep_s21()
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

    // Set a 10 second timeout to account for sweep time.
    viSetAttribute(inst, VI_ATTR_TMO_VALUE, 10e3);

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
    viPrintf(inst, (ViString)":SENSE1:FREQ:START 1GHZ; :SENSE:FREQ:STOP 1GHZ\n");
    // IF bandwidth
    viPrintf(inst, (ViString)":SENSE1:BAND 10KHZ\n");
    // Sweep points
    viPrintf(inst, (ViString)":SENSE1:SWEEP:POINTS %d\n", N);
    // Averaging
    viPrintf(inst, (ViString)":SENSE1:AVERAGE:STATE OFF\n");

    // Delete all traces
    viPrintf(inst, (ViString)":CALC:MEAS:DELETE:ALL\n");

    // Should be 1 plot visible after the preset

    // Setup 4 traces for all of our S-parameters
    // Put them all on plot 1
    // Make them all logMag.

    // Enable trace
    viPrintf(inst, (ViString)":CALC1:MEAS1:DEFINE\n");
    // Set to display 1
    viPrintf(inst, (ViString)":CALC1:MEAS1:DISPLAY 1\n");
    // Set s-parameter
    viPrintf(inst, (ViString)":CALC1:MEAS1:PARAMETER S21\n");
    // Set format (LogMag)
    viPrintf(inst, (ViString)":CALC1:MEAS1:FORMAT MLINEAR\n");

    // Trigger a sweep and wait for it
    int opcResult = 0;
    viQueryf(inst, (ViString)":INIT1:IMM; *OPC?\n", (ViString)"%d", &opcResult);
    assert(opcResult == 1);

    // Query the frequency values for each point
    // These do not change unless the stimulus frequency range is changed.
    std::vector<double> freqs(N);
    int sz = N;
    viQueryf(inst, (ViString)":CALC1:MEAS1:DATA:X?\n", (ViString)"%,#lf", &sz, &freqs[0]);
    assert(sz == N);

    // Query the formatted LinMag data
    std::vector<double> linMag(N);
    sz = N;
    viQueryf(inst, (ViString)":CALC1:MEAS1:DATA:FDATA?\n", (ViString)"%,#lf", &sz, &linMag[0]);
    assert(sz == N);

    // Do something with data here.

    // Done
    viClose(inst);
}
