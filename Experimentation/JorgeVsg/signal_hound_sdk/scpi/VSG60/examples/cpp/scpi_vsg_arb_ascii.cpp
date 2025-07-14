#include <cassert>
#include <cstdio>
#include <complex>
#include <vector>
#include <string>
#include <Windows.h>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to the VSG60 software 
// 2. Configuring the software for ARB waveform generation
// 3. Loading an arbitrary waveform in the ascii format
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

void scpi_vsg_arb_ascii()
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
    viPrintf(inst, "FREQ 1GHz\n");
    viPrintf(inst, "POW -20\n");

    // Create waveform
    // Pulse with N samples, 50% duty cycle
    const int N = 1000000;
    std::vector<std::complex<float>> iq(N);
    for(int i = 0; i < iq.size(); i++) {
        if(i < iq.size() / 2) {
            iq[i].real(1.0);
            iq[i].imag(0.0);
        } else {
            iq[i].real(0.0);
            iq[i].imag(0.0);
        }
    }

    // Configure modulation
    viPrintf(inst, "RADIO:ARB:STATE ON\n");
    viPrintf(inst, "RADIO:ARB:TRIGGER:TYPE CONT\n");
    viPrintf(inst, "RADIO:ARB:SRATE 1MHz\n");

    // Load waveform first
    //std::string asciiWaveform;
    std::stringstream asciiWaveform;
    for(int i = 0; i < iq.size(); i++) {
        asciiWaveform << iq[i].real() << "," << iq[i].imag();
        // Don't add trailing comma
        if(i < iq.size() - 1) {
            asciiWaveform << ",";
        }
    }
    viPrintf(inst, "RADIO:ARB:WAVEFORM:LOAD:IQ:ASCII %s\n", asciiWaveform.str().c_str());

    // We want to output the entire waveform
    viPrintf(inst, "RADIO:ARB:SAMPLE:OFFSET %d\n", 0);
    viPrintf(inst, "RADIO:ARB:SAMPLE:COUNT %d\n", iq.size());

    // Now that waveform is loaded update period to twice the waveform length
    viPrintf(inst, "RADIO:ARB:SAMPLE:PERIOD %d\n", iq.size() * 2);

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