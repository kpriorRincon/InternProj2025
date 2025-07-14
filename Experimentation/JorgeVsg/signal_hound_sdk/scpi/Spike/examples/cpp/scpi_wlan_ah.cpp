#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example  
// 1. Uses VISA to connect to Spike 
// 2. Configures Spike for an 802.11ah measurement.
// 3. Performs N measurements
// 4. Prints the measurement results

void scpi_wlan_ah()
{
    int N = 10;

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
    viPrintf(inst, "WLAN:STANDARD AH\n");
    // Enabled PSDU encoding even though we can't query the bits
    viPrintf(inst, "WLAN:PSDU:DECODE 1\n");
    // -50% symbol offset
    viPrintf(inst, "WLAN:SYMBOL:OFFSET -50\n");
    viPrintf(inst, "WLAN:FREQ:CENT 900MHz\n");
	viPrintf(inst, "WLAN:POW:RF:RLEV -10\n"); 
	viPrintf(inst, "WLAN:IFBW 20MHz\n");

    // Configure the trigger
    viPrintf(inst, "TRIG:WLAN:SLEN 20ms\n");		
    viPrintf(inst, "TRIG:WLAN:IF:THRESHOLD 20\n");		
	
    for(int i = 0; i < N; i++) {
        // Start measurement and wait for trigger
        viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

        // Fetch the results and print them off
        char modType[16];
        char encoding[16];
        char guardInterval[16];
        double evmResults[11];
        viQueryf(inst, ":FETCH:WLAN? 1\n", "%s", modType);
        viQueryf(inst, ":FETCH:WLAN? 2\n", "%s", encoding);
        viQueryf(inst, ":FETCH:WLAN? 3\n", "%s", guardInterval);
        viQueryf(inst, ":FETCH:WLAN? 4,5,6,7,8,9,10,11,12,13,14\n", "%,11lf", evmResults);

        printf("Measurement %d\n", i + 1);
        printf("Modulation Type %s\n", modType);
        printf("Modulation Encoding %s\n", encoding);
        printf("Guard Interval %s\n", guardInterval);
        printf("Bandwidth %f MHz\n", evmResults[10]);
        printf("Freq error %f Hz\n", evmResults[0]);
        printf("Sample rate error %f ppm\n", evmResults[9]);
        printf("EVM %% %f\n", evmResults[1]);
        printf("EVM dB %f\n", evmResults[2]);
        printf("Avg Power dBm %f\n", evmResults[3]);
        printf("Peak Power dBm %f\n", evmResults[4]);
        printf("Crest Factor %f\n", evmResults[5]);
        printf("Initial Scrambler State %f\n", evmResults[6]);
        printf("Symbol Count %f\n", evmResults[7]);
        printf("Payload Bit Count %f\n", evmResults[8]);
        printf("\n\n");
    }

    // Done
    viClose(inst);
}