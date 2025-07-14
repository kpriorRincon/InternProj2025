#include <cassert>
#include <cmath>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to the Spike software
// 2. Configuring a harmonic sweep measurement for the first 10 harmonics
// 3. Performing the sweep
// 4. Fetch the measurement results

// The intended input tone for this example is a modulated signal at 1MHz below -20dBm amplitude

void scpi_harmonic_sweep()
{
    ViSession rm, inst;
    ViStatus rmStatus;

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

    // Get the VISA resource manager
    rmStatus = viOpenDefaultRM(&rm);
    assert(rmStatus == 0);

    // Set the measurement mode to harmonic sweep
    viPrintf(inst, ":INST HARM\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");

    viPrintf(inst, "SENS:HARM:NUMB 10\n"); // 10 harmonics
    viPrintf(inst, ":HARM:TRACK OFF\n"); // Peak tracking off
    viPrintf(inst, ":HARM:MODE PEAK\n"); // Peak measurement mode
    viPrintf(inst, ":HARM:FREQ:FUND 1MHz\n"); // 1MHz fundamental frequency
    viPrintf(inst, ":HARM:FREQ:SPAN 100KHZ\n"); // 100kHz span
    viPrintf(inst, ":HARM:BAND:RES 1kHz; VID 100\n"); // 1kHz RBW, 100Hz VBW
    viPrintf(inst, ":HARM:POW:RF:RLEV -15\n"); // -15dBm input reference level
    viPrintf(inst, ":HARM:VIEW:RLEV -15; PDIV 10\n"); // View at -15dBm reference level, 10dB division
    viPrintf(inst, ":HARM:TRACE:TYPE WRITE\n"); // Clear and write trace type

    // Make sure we have a reasonable timeout value for the sweep to complete. Certain Signal
    // Hound receivers will take long than others to complete a full harmonic sweep.
    viSetAttribute(inst, VI_ATTR_TMO_VALUE, 10e3);

    // Do the sweep, wait for completion
    int opc;
    viQueryf(inst, "INIT; *OPC?\n", "%d", &opc);
    
    // Reset our timeout time
    viSetAttribute(inst, VI_ATTR_TMO_VALUE, 2e3);

    // Fetch measurement results
    printf("Harmonic, Frequency, Amplitude");
    for(int i = 0; i < 10; i++) {
        double freq, ampl;
        viQueryf(inst, ":FETC:HARM:FREQ? %d; AMPL? %d\n", "%lf;%lf", i+1, i+1, &freq, &ampl);
        printf("%d, %.3f MHz, %.2f dBm\n", i+1, freq * 1.0e-6, ampl);
    }

    double distortion = 0.0;
    viQueryf(inst, ":FETC:HARM:DIST?\n", "%lf", &distortion);
    printf("\nTHD %.3f %%\n", distortion);
    printf("%.2f dB\n", 20 * log10(distortion / 100.0));

    // Done
    viClose(inst);
}