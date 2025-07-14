Contact Information: support@signalhound.com

Prerequisites:
The Spike software must be fully installed before using the Python SCPI examples.
Additionally, ensure the Signal Hound device is stable and functioning in Spike.

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
These examples enable the user to send SCPI commands to the Spike software over
a network socket, from a Python programming environment.

All the available SCPI commands are detailed fully in the Spike SCPI Programming
Manual.

Setup:
Run Spike with a Signal Hound device connected.

Usage:
When external commands are received, Spike will display a dialog alerting the
user that it is being controlled remotely. This dialog can be closed, and Spike
operated manually, without interrupting the SCPI connection.
