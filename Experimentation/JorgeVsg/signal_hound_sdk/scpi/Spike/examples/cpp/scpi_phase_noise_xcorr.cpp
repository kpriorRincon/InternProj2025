#include <cassert>
#include <cstdio>
#include <vector>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

#include <Windows.h>

// This example demonstrates 
// 1. Using VISA to connect to Spike
// 2. Setting up a cross correlation measurement. This requires connecting to the
//    PN400 and a second SM series spectrum analyzer.
// 3. Waiting for the measurement to complete.
// 4. Querying some measurement results.

// Intended input signal is a 1GHz CW above the amplitude threshold set in the software
// Spike software should be running with default startup parameters, connected to a single SM device.
// A second SM device should be connected to the PC and inactive.
// The PN400 should also be connected to the PC and inactive.

void scpi_phase_noise_xcorr()
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

    // Several steps in phase noise measurements can take a long time. Set a long timeout
    //   to account for this. 20 seconds
    viSetAttribute(inst, VI_ATTR_TMO_VALUE, 20e3);

    // Select phase noise measurement mode
    viPrintf(inst, "INSTRUMENT:SELECT PN\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");

    // Now setup measurement

    // Signal search
    viPrintf(inst, "PN:CARRIER:SEARCH 1\n");
    viPrintf(inst, "PN:CARRIER:SEARCH:START 500MHz\n");
    viPrintf(inst, "PN:CARRIER:SEARCH:STOP 1.5GHz\n");
    viPrintf(inst, "PN:CARRIER:THR:MIN -40\n");

    // Sweep settings
    viPrintf(inst, "PN:VIEW:RLEV -70; PDIV 10\n");
    viPrintf(inst, "PN:FREQ:CENT 1GHz\n");
    viPrintf(inst, "PN:FREQ:OFFS:STAR 10Hz; STOP 10MHz\n");
    viPrintf(inst, "PN:TYPE PN\n");

    // Setup cross correlation

    // Connect to PN400
    int connected; 
    viQueryf(inst, "PN:VCO:CONNECT?\n", "%d", &connected);
    if(!connected) {
        printf("Unable to connect to PN400\n");
        return;
    }

    // Connect to second SM
    // In this case we know it's an SM200C with a specific address. If it was a USB device,
    //  replace with the serial number of the device rather than the SOCKET string.
    int openSuccess;
    viQueryf(inst, "PN:XCORR:DEVICE:CONNECT? SOCKET::192.168.2.10::51665\n", "%d", &openSuccess);
    if(!openSuccess) {
        // Device failed to open
        printf("Device failed to open\n");
    }

    // Setup the trace, just 1 normal trace
    viPrintf(inst, "TRAC:PN:SEL 1; TYPE NORMAL\n");

    // Now configure cross correlation
    viPrintf(inst, "PN:XCORR:REF INT\n");
    viPrintf(inst, "PN:XCORR:FACTOR 10\n");
    viPrintf(inst, "DISPLAY:PN:XCORR:GINDICATOR ON\n");
    viPrintf(inst, "DISPLAY:PN:XCORR:COUNT ON\n");
    viPrintf(inst, "PN:XCORR ON\n");

    // We now wait for the sweep to complete
    while(true) {
        int progress;
        viQueryf(inst, "PN:XCORR:MEAS:PROGRESS?\n", "%d", &progress);

        if(progress < 10) {
            printf("Progress %d/%d\n", progress, 10);
            Sleep(2000);
        } else {
            printf("Sweep complete\n");
            break;
        }
    }

    // Get decade marker tables
    const int tblSize = 6;
    double offsets[tblSize] = {100, 1e3, 10e3, 100e3, 1e6, 10e6};
    double table[tblSize];

    viPrintf(inst, "CALC:PN:MARKER:SELECT 1");
    viPrintf(inst, "CALC:PN:MARKER ON\n");
    viPrintf(inst, "CALC:PN:MARKER:TRACE 1");

    for(int i = 0; i < tblSize; i++) {
        viPrintf(inst, "CALC:PN:MARK:X %fHz\n", offsets[i]);
        viQueryf(inst, "CALC:PN:MARK:Y?\n", "%lf", table + i);
    }

    // Print it off to the console
    printf("Decade table\n");
    for(int i = 0; i < tblSize; i++) {
        printf("%g Offset: %f dBc\n", offsets[i], table[i]);
    }

    viClose(inst);
}