#include <cstdio>
#include <cstdlib>

#include "sm_api.h"

// This file only applies to SM200C (networked) devices.
// This code assumes the device is connected to a valid 10GbE network
//   interface that has been configured properly (see our NIC setup guides). 
// This example showcases the broadcast configuration function which allows
//   you to modify the IP and port of a given device.
void sm_example_open_with_config_networked_device()
{
    // Host NIC configured IP
    const char *hostAddr = "192.168.2.2";
    // IP and port we want to assign to the SM200
    const char *deviceAddr = "192.168.2.64";
    const uint16_t devicePort = 61111;

    // This function broadcasts a device configuration on the network 
    // interface given by host address. Ideally, the device is connected
    // directly to the PC on this host address. If the device is connected
    // through a switch, since the configuration is sent as a broadcast
    // message, all devices on this interface will see the configuration
    // packet.
    // We specifically make this configuration volatile. This 
    // means when the device is power cycled it will default to its original
    // power on state.
    smBroadcastNetworkConfig(hostAddr, deviceAddr, devicePort, smFalse);

    // Now connect to this device using the new network config.
    int handle = -1;
    SmStatus status = smOpenNetworkedDevice(&handle,
        hostAddr, deviceAddr, devicePort);

    if(status != smNoError) {
        printf("Issue opening device\n");
        printf("%s\n", smGetErrorString(status));
        exit(-1);
    } 

    // At this point the device is open, lets retrieve some
    // information about the device.
    int serial = 0;
    smGetDeviceInfo(handle, 0, &serial);
    printf("Device serial number: %d\n", serial);

    int major, minor, rev;
    smGetFirmwareVersion(handle, &major, &minor, &rev);
    printf("Device firmware version: %d.%d.%d\n", major, minor, rev);

    // Finished
    smCloseDevice(handle);
}