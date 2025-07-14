#include <cassert>
#include <fstream>
#include <vector>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Configuring Spike for digital demodulation measurements of basic PSK signal
// 3. Performing the measurement
// 4. Fetching the demod measurements
// 5. Fetching the constellation plot and saving it to a file

// See configuration below for intended input signal

void scpi_digital_demod_psk_basic()
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
    // Disable continuous meausurement operation and wait for any active measurements
    //  to finish.
    viPrintf(inst, "INIT:CONT OFF\n");

    // Configure the measurement, 1GHz, -20dBm input level
	// QPSK, 1 MHz sym/s, RRC filter
    viPrintf(inst, "DDEMOD:FREQ:CENT 1GHz\n");
	viPrintf(inst, "DDEMOD:POW:RF:RLEV -20DBM\n");
	viPrintf(inst, "DDEMOD:SRAT 1MHz\n");
    viPrintf(inst, "DDEMOD:MOD QPSK\n");		
	viPrintf(inst, "DDEMOD:RLEN 127\n");
	viPrintf(inst, "DDEMOD:FILTER RNYQUIST\n");
	viPrintf(inst, "DDEMOD:FILT:ABT 0.35\n");
	viPrintf(inst, "DDEMOD:IFBW:AUTO ON\n");
	viPrintf(inst, "DDEMOD:AVER ON\n");
	viPrintf(inst, "DDEMOD:AVER:COUN 10\n");
    viPrintf(inst, "TRIG:DDEM:SOUR IMM\n");

    // Do two measurements
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

    // Fetch the results and print them off
    double evmResults[14];
    viQueryf(inst, ":FETCH:DDEMOD? 1,2,3,4,5,6,7,8,9,10,11,12,13,14\n", "%,14lf", evmResults);

    printf("RMS EVM Avg (%%) %f\n", evmResults[0]);
    printf("RMS EVM Peak (%%) %f\n", evmResults[1]);
    printf("\n");

	printf("RMS Mag Err Avg (%%) %f\n", evmResults[2]);
    printf("RMS Mag Err Peak (%%) %f\n", evmResults[3]);
    printf("\n");

	printf("RMS Phase Err Avg (deg) %f\n", evmResults[4]);
    printf("RMS Phase Err Peak (deg) %f\n", evmResults[5]);
    printf("\n");

	printf("IQ Offset Avg (dB) %f\n", evmResults[6]);
    printf("IQ Offset Peak (dB) %f\n", evmResults[7]);
    printf("\n");

	printf("Freq Error Avg (Hz) %f\n", evmResults[8]);
    printf("Freq Error Peak (Hz) %f\n", evmResults[9]);
    printf("\n");

	printf("RF Power Avg (dBm) %f\n", evmResults[10]);
    printf("RF Power Peak (dBm) %f\n", evmResults[11]);
    printf("\n");

	printf("SNR Avg (dB) %f\n", evmResults[12]);
    printf("SNR Peak (dB) %f\n", evmResults[13]);
    printf("\n");

    // Fetch the constellation plot and save it to a CSV
    // Query constellation plot length, as number of complex values
    int constLen;
    viQueryf(inst, "FETCH:DDEMOD? 40\n", "%d", &constLen);
    // constLen is the number of I/Q samples, so we need to request 2 times as many real values
    constLen *= 2; 
    std::vector<float> rdBuffer(constLen);

    // Fetch the constellation plot
    viQueryf(inst, ":FETCH:DDEMOD? 41\n", "%,#f", &constLen, &rdBuffer[0]);

    // Save it to CSV, two columns, I and Q
    std::ofstream file("constellation_plot.csv");
    for(int i = 0; i < constLen/2; i++) {
        file << rdBuffer[i*2] << ", " << rdBuffer[i*2+1] << std::endl;
    }
    file.close();

    printf("Constellation plot saved.\n");

    // Done
    viClose(inst);
}