#pragma once

#define MAX_RF_PORTS 8

typedef enum RfsDeviceType {
    /** RFS8 */
    RfsDeviceTypeRFS8 = 0,
    /** RFS44 */
    RfsDeviceTypeRFS44 = 1
} RfsDeviceType;

typedef enum RfsStatus {
    /** Invalid device handle */
    rfsInvalidDeviceHandle = -6,
    /** Invalid device info */
    rfsInvalidDeviceInfo = -5,
    /** Invalid RF port number */
    rfsInvalidPortNumberErr = -4,
    /** Invalid serial port number */
    rfsInvalidSerialPortNumberErr = -3,
    /** Device disconnected */
    rfsConnectionLostErr = -2,
    /** Unable to open device */
    rfsDeviceNotFoundErr = -1,

    /** Function returned successfully */
    rfsNoError = 0,
} RfsStatus;

RfsStatus rfsOpenDevice(int serialPort, int *device);
RfsStatus rfsCloseDevice(int device);

RfsStatus rfsSetPort(int device, int port);
RfsStatus rfsGetPort(int device, int *port);

RfsStatus rfsQuickSetPort(int serialPort, int port);

RfsStatus rfsGetDeviceInfo(int device, RfsDeviceType *deviceType,
                           unsigned int *sid, unsigned int *firmware);

// Internal use
RfsStatus rfsGetRawHandle(int device, void **handle);
