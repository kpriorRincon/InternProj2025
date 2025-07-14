# Signal Hound RFS8/RFS44 API for Windows and Linux systems

Copyright (c) 2023, Signal Hound, Inc.
For licensing information, please see the API license in the software_licenses folder.

Email/Support: support@signalhound.com

## Overview

This is an API for the Signal Hound RFS Series RF switches, the RFS8 and RFS44.

The API is a set of C functions for connecting to the device's serial port, setting and querying the active RF port, and getting information about the device.

## Setup

The API is provided as source code. To use in C and C++ projects, rfs_api.h and rfs_api.cpp can be included and compiled along with the other source files of an application.

### Windows

The serial port number can be found in Device Manager > Ports (COM & LPT) > USB Serial Port (COM#), where "#" is the integer value of the serial port.

### Linux

The API has been tested on Ubuntu 18.04, and should work on related distributions.

The serial port number can be found with the command `ls /dev`, then locating ttyACM# in the list, where "#" is the integer value of the serial port. This will be 0 if there are no other ACM devices connected.

AppArmor and ModemManager have been known to interfere with the API's communication through the USB serial connection, and prevent proper functioning. They can be disabled with the following commands:

```
sudo systemctl stop apparmor
sudo systemctl disable apparmor

sudo systemctl stop ModemManager.service
sudo systemctl disable ModemManager.service
```

## Usage

All functions return an `RfsStatus` status code, which can be checked for errors. Error codes have negative values. The various codes are listed in the header file.

`rfsOpenDevice` attempts to connect to the serial port passed as a parameter to it, and if successful returns an integer handle. This handle is then used to access the device from the other functions. When operation is finished, `rfsCloseDevice` uses the handle to disconnect the device and clean up resources.

`rfsSetPort` sets the active RF port, that determines where the signal is routed. `rfsGetPort` queries which RF port is currently active.

`rfsQuickSetPort` is a convenience function that opens, sets an RF port, and closes the device.

For repeated switches, it is much faster to open the device once, perform switching, and then close it when done switching. But for one-off switches the convenience function is useful.

Finally, `rfsGetDeviceInfo` returns information about the device, including its serial number, type (RFS8 or RFS44), and hardware/firmware identifier.

## Example

An example program that demonstrates using the API is included: ./api/examples/main.cpp.

### Windows

An IDE/compiler such as Visual Studio can be used to build and run main.cpp against the API source files.

### Linux
A makefile is included to compile the example program.

To build and run the example from within the directory, use the command `make && ./main`.
If you do not have the necessary permissions, you may need to use `make && sudo ./main`.

## GUI

The source code for an example GUI application using the API is included in the ./gui_src/ directory. This requires Qt 5.15.
