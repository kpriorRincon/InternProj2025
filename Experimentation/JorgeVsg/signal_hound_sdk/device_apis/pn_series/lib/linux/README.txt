Signal Hound PN400 API for 64-bit Linux systems

Email/Support: support@signalhound.com

-- Description --
This is the programming interface for the PN400 Signal Hound Phase Noise and VCO Tester for 64-bit Linux.
The API is used for direct control of the device.
The API will function identical to the Windows version, and as such, share examples and documentation. 
Look in the sp_series Windows folder for examples and documentation for this API.

-- Compilation Notes -- 
The API is compiled on
- Ubuntu 18.04 using g++ 7.3.0

-- Installation --
This is a 64 bit shared library. The best way to use the pn_api is to place the included libraries in the /usr/local/lib directory.

Steps to perform this are below.

To install the shared library on your system from the lib folder, type

    sudo cp libpn_api.* /usr/local/lib
    sudo ldconfig -v -n /usr/local/lib
    sudo ln -sf /usr/local/lib/libpn_api.so.1 /usr/local/lib/libpn_api.so
 
This should create the necessary symlinks to the main library and place them in the library directory.

The shared library can now be linked in with g++ by
    g++ sources -o output_exe -Wl, -rpath /usr/local/lib -lpn_api
