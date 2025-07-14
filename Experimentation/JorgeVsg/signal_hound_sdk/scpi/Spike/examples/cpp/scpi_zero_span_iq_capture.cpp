#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Configuring Spike for zero span measurements
// 3. Performing the measurement
// 4. Fetching the resultant IQ data

// The intended input tone for this example is a signal at 1GHz, -20dBm power or less

void scpi_zero_span_iq_capture()
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

    // Set the measurement mode to Zero-Span
    viPrintf(inst, "INSTRUMENT:SELECT ZS\n");
    // Disable continuous meausurement operation and wait for any active measurements
    //  to finish.
    viPrintf(inst, "INIT:CONT OFF\n");

	// Used for scaling
	const double REFLEVEL = -10.0;

    // Configure the measurement, 1GHz, -20dBm input level	
	viPrintf(inst, "SENSE:ZS:CAPTURE:RLEVEL %.2fDBM\n", REFLEVEL);
    viPrintf(inst, "SENSE:ZS:CAPTURE:CENTER 1GHZ\n");
	viPrintf(inst, "SENSE:ZS:CAPTURE:SRATE 50MHZ\n");
	viPrintf(inst, "SENSE:ZS:CAPTURE:IFBW:AUTO OFF\n");	
	viPrintf(inst, "SENSE:ZS:CAPTURE:IFBW 40MHZ\n");	
	viPrintf(inst, "SENSE:ZS:CAPTURE:SWEEP:TIME 0.0015\n"); // sec

    // Do two measurements and wait for them to complete
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);
	viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

	// Use binary format for IQ data
	viPrintf(inst, "FORMAT:IQ:DATA BINARY\n");

	// Query number of points and allocate buffer
    int length;
    viQueryf(inst, "FETCh:ZS? 2\n", "%d", &length);	
	std::vector<ViInt16> rdBuffer(length * 2);

	printf("Length: %d\n", length);

	// Fetch the results and print them off
	ViInt32 rdBufferSize = rdBuffer.size() * sizeof(ViInt16);
	viQueryf(inst, "FETCh:ZS? 1\n", "%#b", &rdBufferSize, rdBuffer.data());

	printf("Length fetched: %d\n", (int)(rdBufferSize / (double)sizeof(ViInt16) / 2.0));

	// Convert from 16-bit ints back to floats	    
	double mwReference = pow(10.0, REFLEVEL / 10.0);
    float scaleFactor = 1.0 / sqrt(mwReference);

	std::vector<float> interleaved(rdBuffer.size());
	for(int i = 0; i < interleaved.size(); i++) {
		interleaved[i] = rdBuffer[i] / 32768.0 / scaleFactor;
	}

	// Convert to AM
	std::vector<float> am(length);
	for(int i = 0; i < am.size(); i++) {
		am[i] = 10.0 * log10(interleaved[i*2] * interleaved[i*2] + interleaved[i*2+1] * interleaved[i*2+1]);
	}

	// Print first 10
	for(int i = 0; i < 10; i++) {
		printf("%.2f\n", am[i]);
	}

    // Done
    viClose(inst);
}