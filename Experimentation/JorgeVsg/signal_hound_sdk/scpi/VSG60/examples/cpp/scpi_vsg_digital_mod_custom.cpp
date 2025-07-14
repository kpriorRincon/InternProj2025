#include <cassert>
#include <cstdio>
#include <Windows.h>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to the VSG60 software 
// 2. Configuring the software for custom digital modulation with user-defined I/Q
// 3. Wait for the *OPC bit to be set in the status register to indicate 
//    the waveform is being output.

static void wait_for_opc(ViSession inst)
{
    viPrintf(inst, "*OPC\n");
    while(true) {
        int esr;
        viQueryf(inst, "*ESR?\n", "%d", &esr);
        if(esr) {
            break;
        }
        Sleep(16);
    }
}

void scpi_vsg_digital_mod_custom()
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

    // Disable modulation
    viPrintf(inst, "OUTPUT ON\n");
    viPrintf(inst, "OUTPUT:MOD OFF\n");

    // Configure freq/ampl
    viPrintf(inst, "FREQ 240MHz\n");
    viPrintf(inst, "POW -30\n");

    // Configure modulation
    viPrintf(inst, "RADIO:CUSTOM ON\n");
    viPrintf(inst, "RADIO:CUSTOM:TRIGGER:TYPE CONTINUOUS\n");
    viPrintf(inst, "RADIO:CUSTOM:SRATE 250kHz\n");
    viPrintf(inst, "RADIO:CUSTOM:MODULATION CUSTOM\n");
	viPrintf(inst, "RADIO:CUSTOM:MODULATION:CUSTOM 1,1,-1,1,-1,-1,1,-1\n");
    viPrintf(inst, "RADIO:CUSTOM:FILTER RNYQUIST\n");
    viPrintf(inst, "RADIO:CUSTOM:FILTER:ALPHA 0.2\n");
    viPrintf(inst, "RADIO:CUSTOM:FILTER:LENGTH 16\n");
    viPrintf(inst, "RADIO:CUSTOM:DATA PN9\n");
    viPrintf(inst, "RADIO:CUSTOM:DATA:SEED 12\n");

    // Enable modulation
    viPrintf(inst, "OUTPUT:MOD ON\n");

    // Wait for operation complete
    wait_for_opc(inst);

    // Make some measurement here
    // ...

    // Done
    viClose(inst);
    return;
}