Copyright (C) Signal Hound, Inc. (2022)

# This library allows you to manually control the TG44 and TG124. 

# It is a Windows DLL, 32-bit build in x86, 64-bit build in x64, compiled with Visual Studio 2019 or 2012.

# FTDI D2XX drivers allow direct access to USB devices through a DLL, and are necessary to control 
the TG. They are available here:
	- https://signalhound.com/sigdownloads/Spike/CDM%20v2.12.00%20WHQL%20Certified.exe

	* This will download ftd2xx DLLs and put them in the system path, but the ones provided here 
	are guaranteed to work with this build. If the system ones aren't working, make sure the 
	provided ftd2xx.dll is in the same directory as the executable.
