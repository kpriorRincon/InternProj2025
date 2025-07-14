#pragma once

#include "../../rfs_api/rfs_api.h"

#include <QObject>

class RFSDevice : public QObject
{
    Q_OBJECT
public:
    RFSDevice();
    ~RFSDevice();

    void Open(int serialPort);
    void Close();

    void SetPort(int port);
    int GetPort();

    bool IsConnected() { return connected; }
    int ActivePort() { return activePort; }
    int Ports();

    RfsDeviceType DeviceType() { return deviceType; }
    unsigned int SerialNumber() { return serialNumber; }
    unsigned int Firmware() { return firmware; }

    QString DeviceName();

private:
    int device;

    bool connected;
    int activePort;

    RfsDeviceType deviceType;
    unsigned int serialNumber;
    unsigned int firmware;

signals:
    void failedToConnect();

    void connectionStateChanged();
    void portChanged();

    void unexpectedDisconnect();
    void invalidPort();
};
