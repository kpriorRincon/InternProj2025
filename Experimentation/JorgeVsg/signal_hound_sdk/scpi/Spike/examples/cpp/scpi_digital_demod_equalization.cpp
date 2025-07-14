#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike.
// 2. Configuring Spike for digital demodulation measurements of a PSK signal.
// 3. Configuring adaptive equalization
// 4. Making several measurement.

// The intended input signal for this example is at
// 1GHz center freq
// -20dBm maximum input power
// BPSK
// 10MSym/s

void scpi_digital_demod_equalization()
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
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");

    // Configure the measurement, 1GHz, -20dBm input level
	// BPSK, 24.3 kHz sample rate, 127 symbol length, root raised cosine filter (alpha 0.35) 
    viPrintf(inst, "DDEMOD:FREQ:CENT 1GHz\n");
	viPrintf(inst, "DDEMOD:POW:RF:RLEV -20DBM\n");
	viPrintf(inst, "DDEMOD:SRAT 10MHz\n");
    viPrintf(inst, "DDEMOD:MOD BPSK\n");		
	viPrintf(inst, "DDEMOD:RLEN 256\n");
	viPrintf(inst, "DDEMOD:FILTER RNYQUIST\n");
	viPrintf(inst, "DDEMOD:FILT:ABT 0.35\n");
	viPrintf(inst, "DDEMOD:IFBW:AUTO ON\n");
	viPrintf(inst, "DDEMOD:AVER OFF\n");
	viPrintf(inst, "DDEMOD:AVER:COUN 1\n");

    // Setup equalization
    viPrintf(inst, "DDEM:EQU ON\n");
    viPrintf(inst, "DDEM:EQU:LENG 15\n");
    viPrintf(inst, "DDEM:EQU:CONV 10.0\n");
    viPrintf(inst, "DDEM:EQU:RESET\n");

    // Perform several measurements allowing the equalizer to adapt printing EVM 
    // on each measurement.
    for(int i = 0; i < 25; i++) {
        viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

        double evm;
        viQueryf(inst, ":FETCH:DDEMOD? 1\n", "%,1lf", &evm);
        printf("EVM (%%) %f\n", evm);
    }

    // Done
    viClose(inst);
}