SM200/SM435 MATLAB Interface Files

These files provide an I/Q interface to the SM product line. 

Functionality:
The SMIQStreamer class provides streaming capabilities for all SM200/SM435 devices.
The class can provide full streaming I/Q up to 50MS/s. For networked devices, 100 and 200
  MS/s rates, single block acquisition is possible, but fully streaming all I/Q data
  into MATLAB is not reasonable.

The SMIQSegmented class provide an interface to the 2 second I/Q capture
  capabilities of the SM200B and SM435B. This I/Q interface provides a
  triggerable I/Q capture. For I/Q streaming, see SMIQStreamer. This interface
  captures I/Q samples at 250MS/s with 160MHz of bandwidth.

Files:
./sm_device/*
  Classes that interface the device through the sm_api.dll. You will use
  this files in your scripts.
streamer_**.m, examples using the streaming I/Q interface.
segmented_**.m, examples using the segmented I/Q interface.

Setup:
Put the sm_api.dll, sm_api.h, files are in the /sm_device/ folder.
You can find these files in the SDK.

Documentation: 
Once the folders are in the MATLAB search path type
'doc SMIQStreamer'  
'doc SMIQSegmented'
for a full description of the classes methods and properties.

Prerequisites:
The Spike software must be fully installed before using the MATLAB 
interface functions. Additionally, ensure the device is stable and 
functioning in Spike before using the MATLAB interface.

Purpose:
These classes serve as a thin wrapper to our SM API, and perform
the majority of C DLL interfacing necessary to communicate with the receiver.

Examples:
To call the functions from outside the /sm_device/ directory you will need
to add the sm_device folder to the MATLAB search path. You can do this by either
using the "Set Path" button on the MATLAB ribbon bar, or finding the /sm_device/
directory in the folder explorer and right clicking and selecting "Add to Path".

To run the test/examples scripts, ensure the /sm_device/ folder has been added to the
MATLAB search path. Navigate to the folder containing the example**.m files.
Each example file is standalone.