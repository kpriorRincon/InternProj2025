#include <cassert>
#include <cstdio>
#include <vector>

#include <Windows.h>

#include "visa.h"
#pragma comment(lib, "visa32.lib")

// This example demonstrates using VISA to connect to Spike, checking for whether
//   a device is currently active after each sweep, and automatically reconnecting
//   after a forced disconnect. 
//
// For this example to be fully illustrative, follow along with the directions printed 
//   to the console. When prompted, disconnect the cable of the Signal Hound device 
//   while running, and then plug it back in a few seconds later to trigger a reconnect.
//
// Note that Spike has an auto-reconnect option, accessed through 
//   Edit > Preferences > General Settings > Connection Settings > Auto Reconnect.
//   This should be turned OFF in Spike for this example, as it accomplishes the same 
//   thing. The difference is this example does it manually, over SCPI.

void setup_device(ViSession inst)
{	
	// Set the measurement mode to Zero-Span
	viPrintf(inst, "INSTRUMENT:SELECT ZS\n");
    // Disable continuous meausurement operation and wait for any active measurements to finish.
    viPrintf(inst, "INIT:CONT OFF\n");
    
	// Configure the measurement, 1GHz, -20dBm input level	
	viPrintf(inst, "SENSE:ZS:CAPTURE:RLEVEL -20DBM\n");
    viPrintf(inst, "SENSE:ZS:CAPTURE:CENTER 1GHZ\n");
	viPrintf(inst, "SENSE:ZS:CAPTURE:IFBW:AUTO ON\n");
	viPrintf(inst, "SENSE:ZS:CAPTURE:SWEEP:TIME 0.0015\n"); // sec
}

void scpi_device_auto_reconnect()
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

	// Configure device in routine so it can be reconfigured after reconnect
	setup_device(inst);

	// Get connection string of current active device so it can be identified 
	// in case there are multiple Signal Hound devices connected to PC.
	char conStr[256];
    viQueryf(inst, "SYST:DEV:CURR?\n", "%s", conStr);
	printf("Connection string for currently active device: %s\n\n", conStr);

	// Sweep with connection check
	// While this loop is running, manually disconnect the device to advance the demonstration
	int activeDevice;
	do {
		// Trigger a sweep, and wait for it to complete
		viQueryf(inst, ":INIT;\n", "%d", &opc);

		// Check that device is still connected
		viQueryf(inst, "SYST:DEV:ACT?\n", "%d", &activeDevice);
		if(activeDevice) printf("Device Active. Manually disconnect to continue demo\n");
	} while(activeDevice == 1);

	// Device has been forcibly disconnected
	printf("\nDevice disconnected\n\n");

	// Spin on reconnect attempts with a two second timeout
	viSetAttribute(inst, VI_ATTR_TMO_VALUE, 2e3);
    int openSuccess = 0;
	while(openSuccess == 0) {
		viQueryf(inst, "SYST:DEVICE:CONNECT? %s\n", "%s", conStr, &openSuccess);
		if(!openSuccess) {
			// Device failed to open
			printf("Waiting for reconnect..\n");
		}
	}
	
	printf("\nDevice reconnected\n\n");

	// Reconfigure device
	printf("Reconfiguring device to previous state\n\n");
	setup_device(inst);
	
	// Begin taking measurements again
	viQueryf(inst, ":INIT;\n", "%d", &opc);
	viQueryf(inst, ":INIT;\n", "%d", &opc);
	viQueryf(inst, ":INIT;\n", "%d", &opc);

	// Done
	printf("Done");
    viClose(inst);	
}