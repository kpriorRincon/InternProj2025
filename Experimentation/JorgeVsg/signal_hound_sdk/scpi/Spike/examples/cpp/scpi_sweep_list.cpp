#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example sweeps several small spans around a fundamental and several
//   harmonics of the input frequency. This code illustrates sweeping different 
//   portions of the spectrum in a loop. 
// This example uses standard sweep measurement mode to accomplish this task. A
//   future update might include the harmonic measurement mode to simplify this task.

// This example demonstrates 
// 1. Using VISA to connect to Spike
// 2. Sweeping several harmonic frequencies of an input tone
// 3. Finding the peak at each harmonic frequency
// 4. Printing the harmonic freq/ampl to the console

void scpi_sweep_list()
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

    double fundamental = 1.0e6; // 1MHz input tone
    int harmonics = 10; // Measure 10 harmonics including the fundmental
    double span = 100.0e3; // 100kHz span around each tone
    double inputLevel = -20.0; // Input CW at -20dBm

    // Some initial one time setup

    // Set the measurement mode to sweep
    viPrintf(inst, "INSTRUMENT:SELECT SA\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");

    // Set the RBW/VBW to auto
    viPrintf(inst, "SENS:BAND:RES:AUTO ON; :BAND:VID:AUTO ON; :BAND:SHAPE FLATTOP\n");
    // Reference level/Div
    viPrintf(inst, "SENS:POW:RF:RLEV %fDBM; PDIV 10\n", inputLevel + 5.0);
    // Average power detector
    viPrintf(inst, "SENS:SWE:DET:FUNC MINMAX; UNIT POWER\n");

    // Configure the trace. Ensures trace 1 is active and enabled for clear/write
    // These commands are not required to be sent everytime, this is for illustrative purposes only. 
    viPrintf(inst, "TRAC:SEL 1\n"); // Select trace 1
    viPrintf(inst, "TRAC:TYPE WRITE\n"); // Set clear and write mode
    viPrintf(inst, "TRAC:UPD ON\n"); // Set update state to on
    viPrintf(inst, "TRAC:DISP ON\n"); // Set un-hidden

    double fundAmpl = 0.0;

    for(int i = 0; i < harmonics; i++) {
        double harmonicFreq = (i+1) * fundamental;

        // Set freq
        viPrintf(inst, "SENS:FREQ:SPAN %f; CENT %f\n", span, harmonicFreq);

        int opc;
        // Trigger a sweep, and wait for it to complete
        viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);
        assert(opc == 1);

        // Perform a peak search, and query the frequency and amplitude
        // If the marker is not currently enabled, then it is enabled on peak search
        viPrintf(inst, "CALC:MARK:MAX; X?; Y?\n");
        double x, y;
        viScanf(inst, "%lf;%lf", &x, &y);

        // Set our fundamental amplitude, so we can calculate dBc for all harmonics
        if(i == 0) {
            fundAmpl = y;
        }

        printf("Marker Peak: %.6f MHz, Ampl: %.2f dBm, Delta: %.2f dBc\n", 
            x/1.0e6, y, y - fundAmpl);
    }

    // Done
    viClose(inst);
}