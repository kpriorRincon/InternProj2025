#include <cassert>
#include <cstdio>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates 
// 1. Using VISA to connect to Spike 
// 2. Configuring Spike for BLE measurement of test transmitter.
// 3. Perform both demod and in-band emission test.
// 4. Fetching the measurement results.

// The intended input tone for this example is a signal 
// - at 2.402GHz, 
// - +10dBm power or less
// - BLE advertising or test packet with 10101010 or 01010101 payload for demod meas
// - PN payload for IBE packet (left to user)

// Modulation characteristics are left to user, will require user to output and measure
//   both the 01010101 and 00001111 patterns to measure all f1/f2 parameters.

void scpi_ble()
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

    // Set the measurement mode to Digital Modulation Analysis
    viPrintf(inst, "INSTRUMENT:SELECT BLE\n");
    // Disable continuous meausurement operation and wait for any active measurements
    //  to finish.
    viPrintf(inst, "INIT:CONT OFF\n");

    // Configure the measurement
    viPrintf(inst, ":BLE:MEAS DEMOD\n");
    viPrintf(inst, ":BLE:FREQ:CENT 2.402GHz\n");
	viPrintf(inst, ":BLE:POW:RF:RLEV +10\n"); // dBm
	viPrintf(inst, ":BLE:IFBW 2MHz\n"); // default
    viPrintf(inst, ":BLE:CHANNEL:AUTO 1\n"); // automatically determine ch. index
    // Configure the trigger
    viPrintf(inst, ":TRIG:BLE:SLEN 20ms\n");		
	
    // Perform 10 measurements
    for(int i = 0; i < 10; i++) {
        viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);
    }

    int powAvg;
    double powMeas[5];
    int cfoAvg;
    double cfoMeas[7];

    // Fetch power demod measurement results
    viQueryf(inst, ":FETCH:BLE? 1\n", "%d", &powAvg);
    viQueryf(inst, ":FETCH:BLE? 2,3,4,5,6\n", "%,5lf", powMeas);
    // Fetch CFO demod measurement results
    viQueryf(inst, ":FETCH:BLE? 200\n", "%d", &cfoAvg);
    viQueryf(inst, ":FETCH:BLE? 201,202,203,204,205,206,207\n", "%,7lf", cfoMeas);

    // Print results
    printf("Output power averages %d\n", powAvg);
    printf("Total avg power %.2f dBm\n", powMeas[0]);
    printf("Max avg power %.2f dBm\n", powMeas[1]);
    printf("Last peak power %.2f dBm\n", powMeas[2]);
    printf("Last avg power %.2f dBm\n", powMeas[3]);
    printf("Last pk-avg power %.2f dBm\n", powMeas[4]);
    printf("\n");
    printf("CFO averages %d\n", cfoAvg);
    printf("Preamble CFO %.3f kHz\n", cfoMeas[0] / 1.0e3);
    printf("Last max CFO %.3f kHz\n", cfoMeas[1] / 1.0e3);
    printf("Last max drift %.3f kHz\n", cfoMeas[2] / 1.0e3);
    printf("Last max drift/50us %.3f kHz\n", cfoMeas[3] / 1.0e3);
    printf("Overall Max CFO %.3f kHz\n", cfoMeas[4] / 1.0e3);
    printf("Overal max drift %.3f kHz\n", cfoMeas[5] / 1.0e3);
    printf("Overal max drift/50us %.3f kHz\n", cfoMeas[6] / 1.0e3);
    printf("\n");

    // Now perform in-band emissions measurement
    // Here is where you would change your transmission to a PN based PDU

    // Config for IBE measurement
    viPrintf(inst, ":BLE:MEAS IBE\n");

    // Do a few measurements if needed for settling
    for(int i = 0; i < 3; i++) {
        viQueryf(inst, ":INIT; *OPC?\n", "%d", &opc);
    }

    double ibeMeas[6];
    viQueryf(inst, ":FETCH:BLE? 300,301,302,303,304,305\n", "%,6lf", ibeMeas);

    printf("IBE result %s\n", ibeMeas[0] ? "Pass" : "Fail");
    printf("IBE Tx channel %.0f\n", ibeMeas[1]);
    printf("IBE pk power %.2f dBm\n", ibeMeas[2]);
    printf("IBE adjacent power (lower) %.2f dBm\n", ibeMeas[3]);
    printf("IBE adjacent power (upper) %.2f dBm\n", ibeMeas[4]);
    printf("IBE failed channels %.0f\n", ibeMeas[5]);

    // Done
    viClose(inst);
}