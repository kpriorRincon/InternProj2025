#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to the VSG60 software 
// 2. Configuring the software to sweep a CW across the frequency range of the device
//    and make a measurement at each frequency.

void scpi_vsg_step_sweep()
{
    ViSession rm, inst;
    ViStatus rmStatus;

    // Get the VISA resource manager
    rmStatus = viOpenDefaultRM(&rm);
    assert(rmStatus == 0);

    // Open a session to the VSG software which must be running at this point
    ViStatus instStatus = viOpen(rm, "TCPIP::localhost::5024::SOCKET", VI_NULL, VI_NULL, &inst);
    assert(instStatus == 0);

    // For SOCKET programming, we want to tell VISA to use a terminating character 
    //   to end a read operation. In this case we want the newline character to end a 
    //   read operation. The termchar is typically set to newline by default. Here we
    //   set it for illustrative purposes.
    viSetAttribute(inst, VI_ATTR_TERMCHAR_EN, VI_TRUE);
    viSetAttribute(inst, VI_ATTR_TERMCHAR, '\n');

    viPrintf(inst, "OUTPUT OFF; :OUTPUT:MOD OFF\n");
    viPrintf(inst, "POW -30\n");

    viPrintf(inst, "STEP:TRIG:TYPE SINGLE\n");
    viPrintf(inst, "STEP:TYPE FREQ\n");
    viPrintf(inst, "STEP:FREQ:START 1GHz; STOP 6GHz\n");
    viPrintf(inst, "STEP:POINTS 1001\n");
    viPrintf(inst, "STEP:DWELL 1ms\n");
    viPrintf(inst, "STEP:STATE ON\n");

    viPrintf(inst, "OUTPUT ON; :OUTPUT:MOD ON\n");

    viClose(inst);

    return;
}