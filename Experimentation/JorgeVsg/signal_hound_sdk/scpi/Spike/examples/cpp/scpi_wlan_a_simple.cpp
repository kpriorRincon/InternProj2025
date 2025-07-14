#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Configuring Spike for an 802.11a measurement.
// 3. Performing the measurement
// 4. Fetching the measurement results.

// The intended input tone for this example is a signal at 1GHz, -20dBm power or less
// The signal should have 802.11a modulation

void scpi_wlan_a_simple()
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

    // Set the measurement mode to WLAN analysis
    viPrintf(inst, "INSTRUMENT:SELECT WLAN\n");
    // Disable continuous meausurement operation and wait for any active measurements
    //  to finish.
    viPrintf(inst, "INIT:CONT OFF\n");

    // Configure the measurement
    viPrintf(inst, "WLAN:STAN AG\n");
    viPrintf(inst, "WLAN:FREQ:CENT 1GHz\n");
	viPrintf(inst, "WLAN:POW:RF:RLEV -10\n"); //dBm
	viPrintf(inst, "WLAN:IFBW 20MHz\n");

    // Configure the trigger
    viPrintf(inst, "TRIG:WLAN:SLEN 20ms\n");		
    viPrintf(inst, "TRIG:WLAN:IF:LEVEL -20\n");		
	
    // Start measurement and wait for trigger
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

    // Fetch the results and print them off
    char modType[16], encoding[16], guardInterval[16];
    double evmResults[9];
    viQueryf(inst, ":FETCH:WLAN? 1\n", "%s", modType);
    viQueryf(inst, ":FETCH:WLAN? 2\n", "%s", encoding);
    viQueryf(inst, ":FETCH:WLAN? 3\n", "%s", guardInterval);
    viQueryf(inst, ":FETCH:WLAN? 4,5,6,7,8,9,10,11,12\n", "%,9lf", evmResults);

    printf("Modulation Type %s\n", modType);
    printf("Modulation Encoding %s\n", encoding);
	printf("Guard Interval %s\n", guardInterval);
    printf("\n");
    printf("Freq error %f\n", evmResults[0]);
	printf("EVM %% %f\n", evmResults[1]);
    printf("EVM dB %f\n", evmResults[2]);
    printf("\n");
	printf("Avg Power dBm %f\n", evmResults[3]);
    printf("Peak Power dBm %f\n", evmResults[4]);
	printf("Crest Factor %f\n", evmResults[5]);
    printf("\n");
    printf("Initial Scrambler State %f\n", evmResults[6]);
	printf("Symbol Count %f\n", evmResults[7]);
    printf("Payload Bit Count %f\n", evmResults[8]);
    printf("\n");

    // Done
    viClose(inst);
}