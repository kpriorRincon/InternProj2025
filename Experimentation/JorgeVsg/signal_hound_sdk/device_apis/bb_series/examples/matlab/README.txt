These instructions were written for 64-bit Windows and will be different on Linux operating systems.

Purpose:
This project enables the user to retrieve sampled I/Q data and sweeps from the BB60C spectrum 
analyzer through a number of convenience functions written in MATLAB. These functions serve 
as a binding to our BB60C API, and perform the majority of C DLL interfacing necessary to communicate
with the receiver.

Prerequisites:
The USB 3.0 drivers must be installed through the Spike software installer.

Setup:
Place the bb_api.dll, bb_api.h, and ftd2xx.dll files into the bb_device/ folder.
Use either the 32-bit or 64-bit DLLs depending on what version of MATLAB you use.
Get the latest API files from the SDK.

To call the functions from outside the bb_device/ directory you will need
to add the bbdevice folder to the MATLAB search path. You can do this by either
using the "Set Path" button on the MATLAB ribbon bar, or finding the bb_device/
directory in the folder explorer and right clicking and selecting
"Add to Path".

To run the test/examples scripts, ensure the bb_device/ folder has been added to the
MATLAB search path. Navigate to the folder containing the bbtest_**.m files.
Each bbtest file is standalone and provides example code for calling the
provided MATLAB functions for the BB60C.
