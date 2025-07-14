#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// It is recommended the user periodically query/flush all errors present in the
//   Spike error queue. This ensures that the user is aware of any issues due to the
//   SCPI command programming of the Spike software.

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Query the number of errors in the queue
// 3. Query each error and print it to the console

void scpi_error_query()
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

    // Lets create a couple errors by sending invalid commands
    for(int i = 0; i < 20; i++) {
        viPrintf(inst, "SENSE:INVALID:COMMAND\n");
    }

    // Query the number of errors present
    int errorCount = 0;
    viQueryf(inst, "SYSTEM:ERROR:COUNT?\n", "%d", &errorCount);
    printf("Error count %d\n", errorCount);

    // For each error, query the error string, print it out in the console
    char errBuf[256];
    for(int i = 0; i < errorCount; i++) {
        viQueryf(inst, "SYST:ERR:NEXT?\n", "%[^\n]", errBuf);
        printf("Error: %s\n", errBuf);
    }

    // Done
    viClose(inst);
}