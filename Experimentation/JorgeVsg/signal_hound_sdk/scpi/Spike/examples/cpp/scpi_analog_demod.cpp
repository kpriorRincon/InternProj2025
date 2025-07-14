#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Configuring Spike for analog demodulation measurements
// 3. Performing the measurement
// 4. Fetching the measurement results.

// The intended input tone for this example is a signal at 1GHz, -20dBm power or less
// The signal ideally is AM/FM modulation but it is not strictly necessary.

void scpi_analog_demod()
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

    // Set the measurement mode to sweep
    viPrintf(inst, "INSTRUMENT:SELECT ADEMOD\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");

    // Configure the measurement, 1GHz, -20dBm input level, 20KHz analog cutoff freq
    viPrintf(inst, "ADEMOD:FREQ:CENT 1GHz\n");
    viPrintf(inst, "ADEMOD:POW:RF:RLEV -20DBM\n");
    viPrintf(inst, "ADEMOD:LPF 20kHz\n");

    // Perform the measurement
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);

    // Fetch the results and print them off
    double amResults[8], fmResults[8];
    viQueryf(inst, ":FETCH:ADEMOD:AM? 1,2,3,4,5,6,7,8\n", "%,8lf", amResults);
    viQueryf(inst, ":FETCH:ADEMOD:FM? 1,2,3,4,5,6,7,8\n", "%,8lf", fmResults);

    printf("Carrier Freq %f\n", amResults[0]);
    printf("Carrier Power %f\n", amResults[1]);
    printf("\n");

    printf("AM Mod Rate %f\n", amResults[2]);
    printf("AM Depth (RMS) %f\n", amResults[3]);
    printf("AM Depth (Peak+) %f\n", amResults[4]);
    printf("AM Depth (Peak-) %f\n", amResults[5]);
    printf("AM SINAD %f\n", amResults[6]);
    printf("AM THD %f\n", amResults[7]);
    printf("\n");

    // Carrier freq/power is the same for FM, skip printing again
    printf("FM Mod Rate %f\n", fmResults[2]);
    printf("FM Depth (RMS) %f\n", fmResults[3]);
    printf("FM Depth (Peak+) %f\n", fmResults[4]);
    printf("FM Depth (Peak-) %f\n", fmResults[5]);
    printf("FM SINAD %f\n", fmResults[6]);
    printf("FM THD %f\n", fmResults[7]);

    // Done
    viClose(inst);
}