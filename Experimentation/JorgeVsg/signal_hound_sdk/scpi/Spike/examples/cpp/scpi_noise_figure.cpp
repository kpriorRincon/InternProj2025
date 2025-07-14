#include <cassert>
#include <cstdio>
#include <Windows.h>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates
// 1. Using VISA to connect to Spike
// 2. Configuring a Noise Figure mode measurement
// 3. Creating a new ENR table for a noise source
// 4. Loading ENR tables for calibration and measurement steps
// 5. Performing a user-interactive calibration
// 6. Performing a user-interactive noise figure and gain measurement
// 7. Waiting for the measurements to complete
// 8. Fetching the measurement results

// This example involves manual connection steps, and provides a basic 
// command line interface to accept user input for progressing through these steps.
// User needs to press return to signal that a setup task has been completed.

// This example requires a powered noise source, such as a Keysight 346B.

void waitForUser() 
{
    printf("Press enter to continue..");

    char response[256];
    while(response[0] != '\n') {
        fgets(response, 255, stdin);
    }
}

void doNextAction(ViSession inst)
{
    char str[4096];

    viQueryf(inst, "STATUS:NFIG:NEXT?\n", "%t", &str);
    printf("\n%s", str);
    waitForUser();
}

void scpi_noise_figure()
{
    ViSession rm, inst;
    ViStatus status;
    int opc;

    char str[4096];

    // Get the VISA resource manager
    status = viOpenDefaultRM(&rm);
    assert(status == 0);

    // Open a session to the Spike software, Spike must be running at this point
    ViStatus instStatus = viOpen(rm, "TCPIP::localhost::5025::SOCKET", VI_NULL, VI_NULL, &inst);
    assert(instStatus == 0);

    // For SOCKET programming, we want to tell VISA to use a terminating character 
    //   to end a read operation. In this case we want the newline character to end a 
    //   read operation. The termchar is typically set to newline by default. Here we
    //   set it for illustrative purposes.
    viSetAttribute(inst, VI_ATTR_TERMCHAR_EN, VI_TRUE);
    viSetAttribute(inst, VI_ATTR_TERMCHAR, '\n');

    // Switch to noise figure mode
    viPrintf(inst, "INSTRUMENT:SELECT NFIGURE\n");

    // Configure measurement
    viPrintf(inst, "NFIG:FREQ:MODE SWEPT\n");
    viPrintf(inst, "NFIG:FREQ:START 10MHz\n");
    viPrintf(inst, "NFIG:FREQ:STOP 3GHZ\n");
    viPrintf(inst, "NFIG:FREQ:POINTS 301\n");

    viPrintf(inst, "NFIG:POWER:RLEVEL -40\n");
    viPrintf(inst, "NFIG:BAND:RES:AUTO ON\n");
    viPrintf(inst, "NFIG:BAND:VID:AUTO ON\n");

    viPrintf(inst, "NFIG:MEAS:SPAN 4MHZ\n");
    viPrintf(inst, "NFIG:AVERAGE ON\n");
    viPrintf(inst, "NFIG:AVERAGE:COUNT 10\n");
    viPrintf(inst, "NFIG:CORR:TCOLD:VAL 290.0\n");

    // Enter a new ENR table 
    viPrintf(inst, "NFIG:CORR:ENR:TABLE:NEW\n");
    int enrTableCount = -1;
    viQueryf(inst, "NFIG:CORR:ENR:TABLE:COUNT?\n", "%d", &enrTableCount);
    viPrintf(inst, "NFIG:CORR:ENR:TABLE:LOAD %d\n", enrTableCount - 1);
    viPrintf(inst, "NFIG:CORR:ENR:TABLE:TITLE Keysight 346B 123456789\n");
    viPrintf(inst, "NFIG:CORR:ENR:TABLE:DATA 10000000,15.45,100e6,15.45,1000e6,15.32,2e9,15.15,3e9,15.09\n");

    // Set new ENR table for noise source of both cal and meas steps
    viPrintf(inst, "NFIG:CORR:ENR:CAL:TABLE %d\n", enrTableCount - 1);
    viPrintf(inst, "NFIG:CORR:ENR:MEAS:TABLE %d\n", enrTableCount - 1);    

    viQueryf(inst, "NFIG:CORR:ENR:TABLE:TITLE?\n", "%t", &str);
    printf("Using Noise Source: %s", str);

    // Begin cal
    printf("\nStarting calibration\n");
    viQueryf(inst, "NFIG:CAL:INIT; *OPC?\n", "%d", &opc);

    // Get next action and wait for user
    doNextAction(inst); // Connect noise source directly to spectrum analyzer

    // Continue
    viQueryf(inst, "NFIG:CONT; *OPC?\n", "%d", &opc);

    // Set timeout to 30 seconds for long sweeps
    viSetAttribute(inst, VI_ATTR_TMO_VALUE, 30e3);

    doNextAction(inst); // Turn noise source on

    // Wait with OPC for sweep to complete
    printf("\nSweeping..\n");
    viQueryf(inst, "NFIG:CONT; *OPC?\n", "%d", &opc);

    doNextAction(inst); // Turn noise source off

    // Get sweep progress intermittently
    viPrintf(inst, "NFIG:CONT\n");
    while(strcmp(str, "100%")) {
        viQueryf(inst, "STATUS:NFIG:PROGRESS?\n", "%t", &str); // Turn noise source off
        str[strcspn(str, "\n")] = 0;
        printf("\nSweep progress: %s", str);
        Sleep(250);
    }
    
    printf("\n\nCalibration complete\n");

    // Begin meas
    printf("\nStarting measurement\n");
    viQueryf(inst, "NFIG:MEAS:INIT; *OPC?\n", "%d", &opc);

    doNextAction(inst); // Connect noise source to DUT to spectrum analyzer
    viQueryf(inst, "NFIG:CONT; *OPC?\n", "%d", &opc);

    doNextAction(inst); // Turn noise source off
    printf("\nSweeping..\n");
    viQueryf(inst, "NFIG:CONT; *OPC?\n", "%d", &opc);

    doNextAction(inst); // Turn noise source on
    printf("\nSweeping..\n");
    viQueryf(inst, "NFIG:CONT; *OPC?\n", "%d", &opc);

    printf("\n\nMeasurement complete\n\n");

    // Done
    viClose(inst);
}
