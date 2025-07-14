#include <cassert>
#include <cstdio>
#include <vector>
#include <string>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates using SCPI to load an already performed calibration and print some of the
//   settings/configuration associated with it.
// This example does not

void scpi_vna_multi_channel()
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



    // Done
    viClose(inst);
}
