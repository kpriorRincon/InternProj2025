#include <cstdio>
#include <cstdlib>

#include "sm_api.h"

// This example illustrates setting the network config of an SM 10GbE
//   device via the USB 2.0 port on the device. 
// The handle used for the network config functions cannot be used for
//   non network config routines.

void sm_example_config_network_via_usb()
{
    // Get list of all networked devices connected to the PC via USB
    int serials[8];
    int deviceCount = 8;
    smNetworkConfigGetDeviceList(serials, &deviceCount);

    if(deviceCount == 0) {
        printf("No networked devices found, aborting.\n");
        return;
    }

    // Open first device found
    int handle = -1;
    SmStatus status = smNetworkConfigOpenDevice(&handle, serials[0]);
    if(status != smNoError) {
        printf("Error opening device, aborting.\n");
        return;
    }

   printf("Opened device with S/N %d.\n", serials[0]);

    char mac[32];
    char ip[32];
    int port;

    // Read settings before applying new configuration
    smNetworkConfigGetMAC(handle, mac);
    smNetworkConfigGetIP(handle, ip);
    smNetworkConfigGetPort(handle, &port);
    printf("Settings prior to configuration\n");
    printf("MAC: %s\n", mac);
    printf("IP: %s\n", ip);
    printf("Port: %d\n", port);

    // Set new network config
    // By setting non-volatile to true, the new settings will persist through
    //  a power cycle. Since a non-volatile configuration involves a flash write,
    //  we recommend limiting how often this occurs. 
    smNetworkConfigSetIP(handle, "192.168.2.10", smTrue);
    smNetworkConfigSetPort(handle, 51665, smTrue);

    // Reread settings to verify new settings
    smNetworkConfigGetMAC(handle, mac);
    smNetworkConfigGetIP(handle, ip);
    smNetworkConfigGetPort(handle, &port);
    printf("Settings after configuration\n");
    printf("MAC: %s\n", mac);
    printf("IP: %s\n", ip);
    printf("Port: %d\n", port);

    smNetworkConfigCloseDevice(handle);
}
