#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike.
// 2. Configuring Spike for digital demodulation measurements of a PSK signal.
// 3. Configuring sync search.
// 4. Making a measurement.

// The intended input signal for this example is at
// 1GHz center freq
// -20dBm maximum input power
// BPSK
// 100kSym/s

void scpi_digital_demod_sync_search()
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
	viPrintf(inst, "DDEMOD:SRAT 100.0kHz\n");
    viPrintf(inst, "DDEMOD:MOD BPSK\n");		
	viPrintf(inst, "DDEMOD:RLEN 256\n");
	viPrintf(inst, "DDEMOD:FILTER RNYQUIST\n");
	viPrintf(inst, "DDEMOD:FILT:ABT 0.35\n");
	viPrintf(inst, "DDEMOD:IFBW:AUTO ON\n");
	viPrintf(inst, "DDEMOD:AVER ON\n");
	viPrintf(inst, "DDEMOD:AVER:COUN 10\n");

    // Setup a sync pattern trigger
    // Trigger on pattern AAAA (16 BPSK symbols)
    // Search length is 1k symbols
    viPrintf(inst, "DDEM:SYNC ON\n");
    viPrintf(inst, "DDEM:SYNC:SWOR:PATT AAAA\n");
    viPrintf(inst, "DDEM:SYNC:SWOR:LENG 16\n");
    viPrintf(inst, "DDEM:SYNC:SLEN 1000\n");
    viPrintf(inst, "DDEM:SYNC:OFFSET 0\n");

    // Perform the measurement
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

    // Read the measured bits and print them off
    char bits[257];
    // Initialize string to zero, pad with 1 char for null termination
    for(int i = 0; i < 257; i++) bits[i] = 0;

    viPrintf(inst, ":FETCH:DDEMOD? 30\n");
    viScanf(inst, "%s", bits);

    printf("Measured bits:\n%s\n", bits);

    // Done
    viClose(inst);
}