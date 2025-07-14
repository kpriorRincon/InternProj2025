#include <cassert>
#include <cstdio>
#include <Windows.h>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Load an LTE preset with scan bands preconfigured
// 3. Perform a single scan
// 4. Wait for the scan to complete
// 5. Pull cell search results

// For this exmaple to work, preset 1 must be setup for LTE measurement mode
//   with at the scan configured. An antenna should be connected to the unit, 
//   and ideally scan bands should be selected that have known LTE cells.

void scpi_lte_scan()
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

    // Load preset for LTE with scan bands
    viPrintf(inst, "*RCL 1\n");
    // As of right now, the software will not respond to commands while the LTE
    //   module is loading.
    // Sleep 15 seconds to ensure the LTE mode has loaded. You can reduce this wait
    //   if you know it loads faster for you, or remove it altogether is you know 
    //   it has already been loaded.
    Sleep(1000 * 15);

    // Disable continuous meausurement operation
    viPrintf(inst, ":INIT:CONT OFF\n");

    // Ensure direct measurements don't get added to cell search results
    viPrintf(inst, ":LTE:MEAS:INCLUDE 0\n");

    // Configure cell search window
    viPrintf(inst, ":LTE:SCAN:TYPE SINGLE\n");
    viPrintf(inst, ":LTE:SCAN:RESULTS:SORT FREQ\n");
    viPrintf(inst, ":LTE:SCAN:RESULTS:KEEP PEAK\n");
    viPrintf(inst, ":LTE:SCAN:RESULTS:GROUP 1\n");
    viPrintf(inst, ":LTE:SCAN:RESULTS:MAX 100\n");

    // Clear any accrued cell search results
    viPrintf(inst, ":LTE:SCAN:RESULTS:CLEAR\n");

    // Perform scan
    int response = 0;
    viQueryf(inst, ":LTE:SCAN:START?\n", "%d", &response);
    // Wait for scan to complete
    while(true) {
        int scanActive = 0;
        viQueryf(inst, ":LTE:SCAN:ACTIVE?\n", "%d", &scanActive);
        if(!scanActive) {
            break;
        } else {
            Sleep(1000);
        }
    }

    // Get number of cell search results
    int cellSearchResults = 0;
    viQueryf(inst, ":LTE:SCAN:RESULTS:COUNT?\n", "%d", &cellSearchResults);

    // Loop through all results and query
    for(int i = 0; i < cellSearchResults; i++) {
        // Set query index into cell search results table
        viPrintf(inst, ":LTE:SCAN:RESULTS:INDEX %d\n", i);

        double freq, earfcn, rssi;
        int physicalCellID, cellID, plmnCount;
        char network[32];
        viQueryf(inst, ":FETCH:LTE? 301,302,303,307,311\n",
            "%lf,%lf,%lf,%d,%d,%d", 
            &freq, &earfcn, &rssi, &physicalCellID, &plmnCount);
        if(plmnCount > 0) {
            viQueryf(inst, ":FETCH:LTE? 315,306\n", "%s,%d",
                network, &cellID);
        }

        printf("Result %d: Freq, %.3f MHz\n", i+1, freq / 1.0e6);
        printf("Result %d: EARFCN, %.2f\n", i+1, earfcn);
        printf("Result %d: RSSI, %.2f dB\n", i+1, rssi);
        printf("Result %d: Phy Cell ID, %d\n", i+1, physicalCellID);
        if(plmnCount > 0) {
            printf("Result %d: Cell ID, %d\n", i+1, cellID);
            printf("Result %d: Network, %s\n", i+1, network);
        }
        printf("\n");
    }

    // Done
    viClose(inst);
}