#include <cassert>
#include <cstdio>
#include <Windows.h>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Put the device into non-continuous mode 
// 3. Configure the device for a direct LTE measurement
// 4. Perform and wait for a measurement to complete
// 5. Request measurement data

// The intended input is an LTE signal at the specified center frequency (751MHz)
// If not valid signal is present, the *OPC? command will timeout.

void scpi_lte_direct_meas()
{
    ViSession rm, inst;
    ViStatus status;

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

    // Extend VISA timeout to ensure we have enough time for the LTE measurement to complete
    viSetAttribute(inst, VI_ATTR_TMO_VALUE, 10e3);

    // Set the measurement mode to sweep
    viPrintf(inst, ":INSTRUMENT:SELECT LTE\n");
    // As of right now, the software will not respond to commands while the LTE
    //   module is loading.
    // Sleep 15 seconds to ensure the LTE mode has loaded. You can reduce this wait
    //   if you know it loads faster for you, or remove it altogether is you know 
    //   it has already been loaded.
    Sleep(1000 * 15);

    // Disable continuous meausurement operation
    viPrintf(inst, ":INIT:CONT OFF\n");

    // Configure a direct measurement
    viPrintf(inst, ":LTE:FREQ:CENTER 751MHz\n");
    viPrintf(inst, ":LTE:POW:RF:RLEVEL -30\n");
    viPrintf(inst, ":LTE:MEAS:INCLUDE 0\n");

    // Perform a single measurement and wait for it to complete
    int opc;
    viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);
    // Make sure we haven't timed out
    assert(status == 0);

    // Get measurement info, RSSI, cellID, network name, and more.
    double rssi, rsrp, rsrq, earfcn;
    int physicalCellID, cellID;
    char bw[32], network[32];
    int sib1Valid;

    viQueryf(inst, ":FETCH:LTE? 5,6,7,100,103,200\n", "%lf,%lf,%lf,%d,%31[^,],%d", 
        &rssi, &rsrp, &rsrq, &physicalCellID, bw, &sib1Valid);

    // If SIB1 was decoded, request data from the SIB1
    if(sib1Valid) {
        viQueryf(inst, ":FETCH:LTE? 201,214,203\n", "%lf,%31[^,],%d",
            &earfcn, network, &cellID);
    }

    // Done
    viClose(inst);
}