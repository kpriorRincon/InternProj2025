#include <cassert>
#include <cstdio>
#include <fstream>
#include <vector>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Configuring Spike for digital demodulation measurements of basic PSK signal
// 3. Performing the measurement
// 4. Fetching the measurement results
// 5. Fetching the frequency vs sample array and saving it to a file

// See configuration below for expected input signal

void scpi_digital_demod_fsk_basic()
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
	// 2FSK, 1MHz sym/s 
    viPrintf(inst, "DDEMOD:FREQ:CENT 1GHz\n");
	viPrintf(inst, "DDEMOD:POW:RF:RLEV -20DBM\n");
	viPrintf(inst, "DDEMOD:SRAT 1MHz\n");
    viPrintf(inst, "DDEMOD:MOD FSK2\n");		
	viPrintf(inst, "DDEMOD:RLEN 127\n");
	viPrintf(inst, "DDEMOD:FILTER GAUS\n");
	viPrintf(inst, "DDEMOD:FILT:ABT 0.5\n");
	viPrintf(inst, "DDEMOD:IFBW:AUTO ON\n");
	viPrintf(inst, "DDEMOD:AVER ON\n");
	viPrintf(inst, "DDEMOD:AVER:COUN 10\n");

    // Do two measurements
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

    // Fetch the results and print them off
    double evmResults[8];
    viQueryf(inst, ":FETCH:DDEMOD? 9,10,11,12,15,16,17,18\n", "%,8lf", evmResults);

    printf("Freq Error Avg (Hz) %f\n", evmResults[0]);
    printf("Freq Error Peak (Hz) %f\n", evmResults[1]);
    printf("\n");

	printf("RF Power Avg (dBm) %f\n", evmResults[2]);
    printf("RF Power Peak (dBm) %f\n", evmResults[3]);
    printf("\n");

    printf("RMS FSK Avg (%%) %f\n", evmResults[4]);
    printf("RMS FSK Peak (%%) %f\n", evmResults[5]);
    printf("\n");

	printf("FSK Dev Avg (Hz) %f\n", evmResults[6]);
    printf("FSK Dev Peak (Hz) %f\n", evmResults[7]);
    printf("\n");

    // Fetch the constellation/frequency plot and save it to a CSV
    // Query plot length
    int plotLen;
    viQueryf(inst, "FETCH:DDEMOD? 40\n", "%d", &plotLen);
    // constLen is the number of I/Q samples, so we need to request 2 times as many real values
    std::vector<float> rdBuffer(plotLen);

    // Fetch the constellation plot
    viQueryf(inst, ":FETCH:DDEMOD? 41\n", "%,#f", &plotLen, &rdBuffer[0]);

    // Save it to CSV
    std::ofstream file("freq_vs_time_plot.csv");
    for (int i = 0; i < plotLen; i++) {
        file << rdBuffer[i] * evmResults[6] << std::endl;
    }
    file.close();

    printf("Plot saved.\n");

    // Done
    viClose(inst);
}