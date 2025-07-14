#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Configuring Spike for digital demodulation measurements using a custom modulation

void scpi_digital_demod_custom_modulation()
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

    // Set the measurement mode to Digital Modulation Analysis
    viPrintf(inst, "INSTRUMENT:SELECT DDEMOD\n");

	// Configure custom modulation
	viPrintf(inst, "DDEM:CUST:IQ:DATA 1,1,-1,1,-1,-1,1,-1\n");
	viPrintf(inst, "DDEMOD:MOD CUSTom\n");

    // Done
    viClose(inst);
}