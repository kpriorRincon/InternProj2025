#include "rfs_device.h"

RFSDevice::RFSDevice()
{
    connected = false;
    activePort = 1;
    serialNumber = 0;
    firmware = 0;
}

RFSDevice::~RFSDevice()
{
    Close();
}

void RFSDevice::Open(int serialPort)
{
    if(connected) {
        return;
    }

    if(rfsOpenDevice(serialPort, &device) != rfsNoError) {
        emit failedToConnect();
        return;
    }

    connected = true;

    rfsGetDeviceInfo(device, &deviceType, &serialNumber, &firmware);

    emit connectionStateChanged();

    SetPort(1);
}

void RFSDevice::Close()
{
    if(connected) {
        rfsCloseDevice(device);
        connected = false;
        emit connectionStateChanged();
    }
}

void RFSDevice::SetPort(int port)
{
    if(!connected) {
        emit unexpectedDisconnect();
        return;
    }

    if(port < 1 || port > Ports()) {
        emit invalidPort();
        return;
    }

    if(rfsSetPort(device, port) != rfsNoError) {
        emit unexpectedDisconnect();
        connected = false;
        return;
    }

//    // Check port
//    int newPort = -1;
//    if(rfsGetPort(device, &newPort) != rfsNoError
//            || newPort != port) {
//        emit unexpectedDisconnect();
//        connected = false;
//        return;
//    }

    activePort = port;

    emit portChanged();
}

int RFSDevice::GetPort()
{
    if(!connected) {
        emit unexpectedDisconnect();
        return -1;
    }

    int port;
    if(rfsGetPort(device, &port) != rfsNoError) {
        emit unexpectedDisconnect();
        connected = false;
        return -1;
    }

    return port;
}

int RFSDevice::Ports()
{
    switch(deviceType) {
    case RfsDeviceTypeRFS8:
        return 8;
    case RfsDeviceTypeRFS44:
        return 4;
    default:
        // Invalid device type
        return 0;
    }
}

QString RFSDevice::DeviceName()
{
    switch(deviceType) {
    case RfsDeviceTypeRFS8:
        return "RFS8";
    case RfsDeviceTypeRFS44:
        return "RFS44";
    default:
        // Invalid device type
        return "";
    }
}
