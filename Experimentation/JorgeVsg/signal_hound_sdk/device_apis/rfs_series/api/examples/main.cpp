#include "../src/rfs_api.h"

#ifdef _WIN32 // Windows
#define DEFAULT_SERIAL_PORT 1
#else // Linux
#define DEFAULT_SERIAL_PORT 0
#endif

int main()
{
    // Change this from default to actual port number (see README for details)
    const int serialPort = DEFAULT_SERIAL_PORT;

    // Convenience function to quickly set port
    rfsQuickSetPort(serialPort, 1);

    // For greater speed when switching ports repeatedly
    int device;

    rfsOpenDevice(serialPort, &device);

    rfsSetPort(device, 1);
    rfsSetPort(device, 2);
    rfsSetPort(device, 3);

    rfsCloseDevice(device);

    return 0;
}
