#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Configuring Spike for spectrum emission mask measurements 
//		- Includes defining a mask by loading data into an offset table
// 3. Performing the measurement
// 4. Fetching the measurement results

// The intended input tone for this example is a signal at 1GHz, -20dBm power or less

void scpi_spectrum_emission_mask_basic()
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

    // Set the measurement mode to Spectrum Emission Mask
    viPrintf(inst, "INSTRUMENT:SELECT SEM\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");

    // Configure the measurement
	// 1GHz, -20dBm input level, channel power measurement type
	// Manually input Bluetooth mask into offset table
    
	// Frequency
	viPrintf(inst, "SEM:FREQ:CENT 1GHz\n");
	viPrintf(inst, "SEM:FREQ:CENT:STEP:INCR 10MHz\n");
	viPrintf(inst, "SEM:FREQ:SPAN 10MHz\n");

	// Bandwidth
	viPrintf(inst, "SEM:BAND:RES:AUTO ON\n");
	viPrintf(inst, "SEM:BAND:VID:AUTO ON\n");

	// Amplitude
	viPrintf(inst, "SEM:POW:RF:RLEV -20\n");
	viPrintf(inst, "SEM:POW:RF:PDIV 10\n");

	// Detector / Trace
	viPrintf(inst, "SEM:SWE:DET:UNIT POW\n");
	viPrintf(inst, "SEM:SWE:DET:FUNC AVER\n");
	viPrintf(inst, "TRAC:SEM:TYPE WRITE\n");

	// Measurement Reference
	viPrintf(inst, "SEM:REF:TYPE PSD\n");
	viPrintf(inst, "SEM:REF:BANDwidth:MODE AUTO\n");
	
	// Load mask into offset table (Bluetooth)
	viPrintf(inst, "SEM:OFFS:DATA ON, 1MHz, 1.5MHz, -26, -26, REL, ON, 1.5MHz, 2.5MHz, -46, -46, ABS, ON, 2.5MHz, 5MHz, -86, -86, ABS\n");

    // Do two measurements
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

    // Get carrier / reference power
    double carrierPower = 0;
    viQueryf(inst, "SEM:CARR:POW?\n", "%lf", &carrierPower);

	printf("Carrier (Reference) Power: %f dBm\n", carrierPower);
    printf("\n");

	// Test against mask and print
	int fail = false;
	viQueryf(inst, "SEM:OFFSET:FAIL?\n", "%d", &fail);

	printf((bool)fail ? "Mask failed\n" : "Mask passed\n");
    printf("\n");

	// Find lower frequency, level, and margin (limit - peak) of offset 2
	double lowerFreq = 0, lowerLevel = 0, lowerMargin = 0;
	viQueryf(inst, "SEM:OFFSET2:PEAK:FREQ:LOWER?\n", "%lf", &lowerFreq);
	viQueryf(inst, "SEM:OFFSET2:PEAK:LEVEL:LOWER?\n", "%lf", &lowerLevel);
	viQueryf(inst, "SEM:OFFSET2:MARGIN:LOWER?\n", "%lf", &lowerMargin);

	printf("Offset 2:\n\tPeak Freq: %.2f MHz\n\tPeak Level: %.2f dBm\n\tMargin (Limit - Peak): %.2f dB\n", lowerFreq/1e6, lowerLevel, lowerMargin);
    printf("\n");

    // Done
    viClose(inst);
}