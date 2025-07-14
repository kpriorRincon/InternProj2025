#include <cassert>
#include <cstdio>
#include <vector>
#include <string>
#include <complex>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates using SCPI to read the raw s-parameter data from the instrument
// This example sets up a full 2 port sweep, and reads all 4 s-parameters.
// This example does not include the use of a SOL calibration.

void scpi_vna_read_sparams_2port()
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
    viPrintf(inst, (ViString)":SENSE1:FREQ:START 40MHZ; :SENSE:FREQ:STOP 40GHZ\n");
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
    viPrintf(inst, (ViString)":CALC1:MEAS1:PARAMETER S11\n");
    // Set format (LogMag)
    viPrintf(inst, (ViString)":CALC1:MEAS1:FORMAT MLOG\n");

    // Now repeat for the other 3 s-parameters/traces

    // S21
    viPrintf(inst, (ViString)":CALC1:MEAS2:DEFINE\n");
    viPrintf(inst, (ViString)":CALC1:MEAS2:DISPLAY 1\n");
    viPrintf(inst, (ViString)":CALC1:MEAS2:PARAMETER S21\n");
    viPrintf(inst, (ViString)":CALC1:MEAS2:FORMAT MLOG\n");
    // S12
    viPrintf(inst, (ViString)":CALC1:MEAS3:DEFINE\n");
    viPrintf(inst, (ViString)":CALC1:MEAS3:DISPLAY 1\n");
    viPrintf(inst, (ViString)":CALC1:MEAS3:PARAMETER S12\n");
    viPrintf(inst, (ViString)":CALC1:MEAS3:FORMAT MLOG\n");
    // S22
    viPrintf(inst, (ViString)":CALC1:MEAS4:DEFINE\n");
    viPrintf(inst, (ViString)":CALC1:MEAS4:DISPLAY 1\n");
    viPrintf(inst, (ViString)":CALC1:MEAS4:PARAMETER S22\n");
    viPrintf(inst, (ViString)":CALC1:MEAS4:FORMAT MLOG\n");

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

    // Now query the s-parameter data
    // Returned as complex data, 2 values (I/Q) for each freq point.

    // Arrays for all four sparameters in a loop
    std::vector<std::complex<double>> sparam[4];
    for(int i = 0; i < 4; i++) {
        // Allocate memory
        sparam[i].resize(N);
        // Tell VISA to look for N*2 points, since complex.
        sz = N*2;
        viQueryf(inst, (ViString)":CALC1:MEAS%d:DATA:SDATA?\n", (ViString)"%,#lf", i+1, &sz, &sparam[i][0]);
        // Verify we have received all the data.
        assert(sz == N*2);
    }

    // Do something with data here.

    // Done
    viClose(inst);
}
