Contact Information: support@signalhound.com

Prerequisites:
The Signal Hound VSG60 Vector Signal Generator software must be fully installed
before using the Python SCPI examples. Additionally, ensure the VSG60 is stable
and functioning in the VSG60 software.

Python 3 is required to run the Python SCPI examples.
They have been tested with Python 3.6.1 on Windows.

PyVISA, along with its dependencies, must be installed.
For more information, visit https://pyvisa.readthedocs.io/.
PyVISA is a frontend for a VISA library such as National Instruments' VISA and
Keysight IO Library Suite.
A backend library such as these must be installed. For more information, visit:
    https://www.ni.com/en-us/support/downloads/drivers/download.ni-visa.html
    https://www.keysight.com/us/en/lib/software-detail/computer-software/io-libraries-suite-downloads-2175637.html

Purpose:
These examples enable the user to send SCPI commands to the VSG60 software over
a network socket, from a Python programming environment.

All the available SCPI commands are detailed fully in the VSG60 SCPI Programming
Manual.

Setup:
Run the VSG60 software with a VSG60 device connected.

Usage:
When external commands are received, the VSG60 software will display a dialog
alerting the user that it is being controlled remotely. This dialog can be closed,
and the software operated manually, without interrupting the SCPI connection.
