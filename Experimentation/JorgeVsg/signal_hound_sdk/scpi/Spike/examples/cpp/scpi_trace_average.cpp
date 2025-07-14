#include <cassert>
#include <cstdio>
#include <vector>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike
// 2. Configuring an average trace
// 3. Performing several sweeps to generate the avg trace
// 4. Querying the trace data
// 5. Manual peak search on the trace data and determine freq,ampl of peak

// Intended input signal is 1GHz less than -20dBm

void scpi_trace_average()
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

    // We want to create an average trace of 20 sweeps
    int averageCount = 20;

    // Set the measurement mode to sweep
    viPrintf(inst, "INSTRUMENT:SELECT SA\n");
    // Disable continuous meausurement operation
    viPrintf(inst, "INIT:CONT OFF\n");

    // Configure a 20MHz span sweep at 1GHz
    // Set the RBW/VBW to auto
    viPrintf(inst, "SENS:BAND:RES:AUTO ON; :BAND:VID:AUTO ON; :BAND:SHAPE FLATTOP\n");
    // Center/span
    viPrintf(inst, "SENS:FREQ:SPAN 20MHZ; CENT 1GHZ\n");
    // Reference level/Div
    viPrintf(inst, "SENS:POW:RF:RLEV -30DBM; PDIV 10\n");
    // Average power detector
    viPrintf(inst, "SENS:SWE:DET:FUNC AVERAGE; UNIT POWER\n");

    // Configure the trace. Ensures trace 1 is active and enabled for averaging
    // These commands are not required to be sent everytime, this is for illustrative purposes only. 
    viPrintf(inst, "TRAC:SEL 1\n"); // Select trace 1
    viPrintf(inst, "TRAC:TYPE AVERAGE\n"); // Set clear and write mode
    viPrintf(inst, "TRACE:AVER:COUNT %d\n", averageCount);
    viPrintf(inst, "TRAC:UPD ON\n"); // Set update state to on
    viPrintf(inst, "TRAC:DISP ON\n"); // Set un-hidden
    viPrintf(inst, "TRACE:CLEAR\n"); // Clear it

    // Perform 'averageCount' sweeps
    for(int i = 0; i < averageCount; i++) {
        int opc;
        // Trigger a sweep, and wait for it to complete
        viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);
        assert(opc == 1);
    }

    // Now lets get the full sweep
    int traceLength;
    // First query the number of points in the sweep
    viQueryf(inst, "TRACE:POINTS?\n", "%d", &traceLength);
    // Preallocate array for the sweep
    traceLength *= 2;
    std::vector<float> points(traceLength);
    // Ask for the sweep data
    // Sweep data is returned as comma separated values
    viQueryf(inst, "TRACE:DATA?\n", "%,#f", &traceLength, &points[0]);

    // Query information needed to know what frequency each point in the sweep refers to
    double xStart, xInc;
    viQueryf(inst, "TRACE:XSTART?; XINC?\n", "%lf;%lf\n", &xStart, &xInc);

    // Find the peak point in the sweep
    int peakIx = 0;
    float peakVal = points[0];
    for(int i = 1; i < traceLength; i++) {
        if(points[i] > peakVal) {
            peakIx = i;
            peakVal = points[i];
        }
    }

    // Print out peak information
    printf("Peak Freq %f MHz, Peak Ampl %f dBm\n",
        (xStart + xInc * peakIx) / 1.0e6, peakVal);

    // Done
    viClose(inst);
}