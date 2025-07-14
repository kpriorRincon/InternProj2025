#include "rfs_api.h"

#include <stdio.h>
#include <math.h>

#ifdef _WIN32 // Windows
#include <Windows.h>
#else // Linux
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <termios.h>
#include <string.h>
#endif

#define SERIAL_PORT_MIN 0
#define SERIAL_PORT_MAX 255

#define RFS8_HWFW  0x2020202
#define RFS44_HWFW 0x1010101

#define MAX_DEVICES 8
static void* deviceList[MAX_DEVICES];
static int nextDevice = 0;

#define VALID_INDEX_CHECK(id) \
    if(id < 0 || id >= MAX_DEVICES) { return rfsInvalidDeviceHandle; }

RfsStatus rfsOpenDevice(int serialPort, int *device)
{
    VALID_INDEX_CHECK(nextDevice);

#ifdef _WIN32 // Windows
    if((serialPort < SERIAL_PORT_MIN) || (serialPort > SERIAL_PORT_MAX)) {
        return rfsInvalidSerialPortNumberErr;
    }

    char fileString[16];
    sprintf_s(fileString, 16, "COM%d", serialPort);

    HANDLE handle = CreateFileA(fileString, GENERIC_READ | GENERIC_WRITE,
                                0, 0, OPEN_EXISTING, 0, 0);

    if(handle == INVALID_HANDLE_VALUE) {
        return rfsDeviceNotFoundErr;
    }
#else // Linux
    if((serialPort < SERIAL_PORT_MIN) || (serialPort > SERIAL_PORT_MAX)) {
        return rfsInvalidSerialPortNumberErr;
    }

    char fileString[16];
    snprintf(fileString, sizeof(fileString), "/dev/ttyACM%d", serialPort);

    int _handle = open(fileString, O_RDWR);

    if(_handle < 0) {
        return rfsDeviceNotFoundErr;
    }

    struct termios tty;
    if(tcgetattr(_handle, &tty) != 0) {
        return rfsDeviceNotFoundErr;
    }

    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;
    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tty.c_lflag &= ~ECHOE;
    tty.c_lflag &= ~ECHONL;
    tty.c_lflag &= ~ISIG;

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);

    tty.c_oflag &= ~OPOST;
    tty.c_oflag &= ~ONLCR;

    tty.c_cc[VTIME] = 10;
    tty.c_cc[VMIN] = 0;

    cfsetispeed(&tty, B9600);
    cfsetospeed(&tty, B9600);

    if (tcsetattr(_handle, TCSANOW, &tty) != 0) {
        return rfsDeviceNotFoundErr;
    }

    int *handle = (int*)malloc(sizeof(int));
    *handle = _handle;
#endif

    *device = nextDevice;
    deviceList[nextDevice++] = handle;

    return rfsNoError;
}

RfsStatus rfsCloseDevice(int device)
{
    VALID_INDEX_CHECK(device);

#ifdef _WIN32 // Windows
    HANDLE handle = deviceList[device];

    if(CloseHandle(handle) == 0) {
        return rfsConnectionLostErr;
    }
#else // Linux
    int handle = *(int*)deviceList[device];

    if(close(handle) != 0) {
        return rfsConnectionLostErr;
    }

    free(deviceList[device]);
#endif

    return rfsNoError;
}

RfsStatus rfsSetPort(int device, int port)
{
    VALID_INDEX_CHECK(device);

    if((port < 1) || (port > MAX_RF_PORTS)) {
        return rfsInvalidPortNumberErr;
    }

    char cmd[2];
    sprintf(cmd, "%d", port);

#ifdef _WIN32 // Windows
    HANDLE handle = deviceList[device];

    const DWORD byteCount = sizeof(cmd);
    DWORD bytesWritten = 0;
    if(!WriteFile(handle, &cmd, byteCount, &bytesWritten, NULL)) {
        return rfsConnectionLostErr;
    }

    if(bytesWritten < byteCount) {
        return rfsConnectionLostErr;
    }
#else // Linux
    int handle = *(int*)deviceList[device];

    const ssize_t byteCount = sizeof(cmd);
    const ssize_t bytesWritten = write(handle, &cmd, byteCount);
    if(bytesWritten < byteCount) {
        return rfsConnectionLostErr;
    }

    if(tcdrain(handle)) {
        return rfsConnectionLostErr;
    }
#endif

    return rfsNoError;
}

RfsStatus rfsGetPort(int device, int *port)
{
    VALID_INDEX_CHECK(device);

#ifdef _WIN32 // Windows
    HANDLE handle = deviceList[device];

    BYTE cmd = 'P';
    const DWORD writeCount = sizeof(cmd);
    DWORD bytesWritten = 0;
    if(!WriteFile(handle, &cmd, writeCount, &bytesWritten, NULL)) {
        return rfsConnectionLostErr;
    }

    if(bytesWritten < writeCount) {
        return rfsConnectionLostErr;
    }

    Sleep(100);

    char buf[2];
    const DWORD readCount = sizeof(buf);
    DWORD bytesRead = 0;
    if(!ReadFile(handle, buf, readCount, &bytesRead, NULL)) {
        return rfsConnectionLostErr;
    }

    if(bytesRead < readCount) {
        return rfsConnectionLostErr;
    }
#else // Linux
    int handle = *(int*)deviceList[device];

    unsigned char cmd[] = {'P'};
    const ssize_t writeCount = sizeof(cmd);
    const ssize_t bytesWritten = write(handle, cmd, writeCount);

    if(bytesWritten < writeCount) {
        return rfsConnectionLostErr;
    }

    usleep(100e3);

    char buf[2];
    const ssize_t readCount = sizeof(buf);
    ssize_t bytesRead = read(handle, buf, readCount);

    if(bytesRead < readCount) {
        return rfsConnectionLostErr;
    }
#endif

    int p = atoi(&buf[1]);

    if((p < 1) || (p > MAX_RF_PORTS)) {
        return rfsInvalidPortNumberErr;
    }

    if(port) *port = p;

    return rfsNoError;
}

RfsStatus rfsQuickSetPort(int serialPort, int port)
{
    int device;
    RfsStatus status;

    status = rfsOpenDevice(serialPort, &device);
    if(status != rfsNoError) {
        return status;
    }

    status = rfsSetPort(device, port);
    if(status != rfsNoError) {
        return status;
    }

    return rfsCloseDevice(device);
}

RfsStatus rfsGetDeviceInfo(int device, RfsDeviceType *deviceType,
                           unsigned int *sid, unsigned int *firmware)
{
    VALID_INDEX_CHECK(device);

#ifdef _WIN32 // Windows
    HANDLE handle = deviceList[device];

    BYTE cmd[4] = {'R', 'f', 'h', 's'};
    const DWORD writeCount = sizeof(cmd);
    DWORD bytesWritten = 0;
    if(!WriteFile(handle, cmd, writeCount, &bytesWritten, NULL)) {
        return rfsConnectionLostErr;
    }

    if(bytesWritten < writeCount) {
        return rfsConnectionLostErr;
    }

    Sleep(100);

    unsigned int buf[3];
    const DWORD readCount = sizeof(buf);
    DWORD bytesRead = 0;
    if(!ReadFile(handle, buf, readCount, &bytesRead, NULL)) {
        return rfsConnectionLostErr;
    }

    if(bytesRead < readCount) {
        return rfsConnectionLostErr;
    }
#else // Linux
    int handle = *(int*)deviceList[device];

    unsigned char cmd[] = {'R', 'f', 'h', 's'};
    const ssize_t writeCount = sizeof(cmd);
    ssize_t bytesWritten = write(handle, cmd, writeCount);

    if(bytesWritten < writeCount) {
        return rfsConnectionLostErr;
    }

    usleep(100e3);

    unsigned int buf[3];
    const ssize_t readCount = sizeof(buf);
    const ssize_t bytesRead = read(handle, buf, readCount);

    if(bytesRead < readCount) {
        return rfsConnectionLostErr;
    }
#endif

    const unsigned int serial = buf[1];
    if(int(log10(serial) + 1) != 8) {
        return rfsInvalidDeviceInfo;
    }

    const unsigned int hwfw = buf[2];
    if(hwfw != RFS8_HWFW && hwfw != RFS44_HWFW) {
        return rfsInvalidDeviceInfo;
    }

    if(deviceType) *deviceType = hwfw & 1 << 8 ? RfsDeviceTypeRFS44 : RfsDeviceTypeRFS8;
    if(sid) *sid = serial;
    if(firmware) *firmware = hwfw;

    return rfsNoError;
}

RfsStatus rfsGetRawHandle(int device, void **handle)
{
    VALID_INDEX_CHECK(device);

    *handle = deviceList[device];

    return rfsNoError;
}
