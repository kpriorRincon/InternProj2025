#include <cassert>
#include <cstdio>
#include <vector>
#include <string>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates using SCPI to load a calibration file 
//   and printing some of the settings/configuration associated with it.
// This example does not perform any measurements.

void scpi_vna_load_cal()
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

    // Load a calibration file.
    // Replace the file path + name with your own file.
    // Set 1 to load stimulus.
    viPrintf(inst, (ViString)":SENSE1:CORRECTION:CSET:ACTIVATE \"C:/Users/AJ/Documents/SignalHound/vna/cals/example_cal.vnacal\", 1\n");

    // Verify it loaded, query some properties.
    char calSetName[64] = "";
    viQueryf(inst, (ViString)":SENSE1:CORRECTION:CSET:NAME?\n", (ViString)"%[^\n]", calSetName);

    char calSetDate[64] = "";
    viQueryf(inst, (ViString)":SENSE1:CORRECTION:CSET:DATE?\n", (ViString)"%[^\n]", calSetDate);

    printf("Calibration Name: %s\n", calSetName);
    printf("Calibration Date: %s\n", calSetDate);

    // Query stimulus
    double startFreq;
    double stopFreq;
    double ifBandwidth;
    int points;
    viQueryf(inst, (ViString)":SENSE1:FREQ:START?\n", (ViString)"%lf", &startFreq);
    viQueryf(inst, (ViString)":SENSE1:FREQ:STOP?\n", (ViString)"%lf", &stopFreq);
    viQueryf(inst, (ViString)":SENSE1:BANDWIDTH?\n", (ViString)"%lf", &ifBandwidth);
    viQueryf(inst, (ViString)":SENSE1:SWEEP:POINTS?\n", (ViString)"%d", &points);

    printf("Start Freq: %f\n", startFreq);
    printf("Stop Freq: %f\n", stopFreq);
    printf("IF Bandwidth: %f\n", ifBandwidth);
    printf("Sweep Points: %d\n", points);

    // Done
    viClose(inst);
}
