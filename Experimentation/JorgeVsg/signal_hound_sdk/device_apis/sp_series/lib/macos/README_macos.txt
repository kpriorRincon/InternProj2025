Signal Hound SP145 Programming API for MacOSX

Email/Support: support@signalhound.com

-- Documentation -- 
https://signalhound.com/sigdownloads/SDK/online_docs/sp_api/index.html

-- Limitations --
The follow limitations apply to the ARM build of the API,
- The API is limited to the sweep and I/Q streaming modes only.
- In sweep mode, RBW and VBW is limited to 100Hz.
- In sweep mode, the sweep speed is limited to ~80GHz/s instead of the normal 200GHz/s.
- In I/Q streaming mode, decimation is limited to values of 1, 2, and 4.
- In I/Q streaming mode, the software filter cannot be disabled. This means there can be
  spurious and aliased signals in the rejection regions. The passband will not be affected. 
  If spurious signals in the reject bands are not acceptable, we recommend
  running an additional post acquisition FIR filter.

-- Compilation Notes -- 
The API is compiled on a Macbook Pro
- M4 Pro CPU
- Sequioa 15.1
- Apple clang++ 16.0.0

-- Libusb 1.0 requirement -- 
The API depends on the libusb-1.0 USB drivers.
You will need libusb-1.0 installed in the system path. 
You can install libusb-1.0 with homebrew.

-- Device Permissions --
To run an application utilizing libusb, you need to be have root permissions. 
You can either run your application as root or change permissions for the device.

-- Installation (Linux) --
This is a 64 bit shared library. The best way to use the sp_api is to place the included libraries in the /usr/local/lib directory. 

Steps to perform this are below.

To install the shared library on your system from the lib folder, type

    sudo cp libsp_api.* /usr/local/lib
    sudo ldconfig -v -n /usr/local/lib
    sudo ln -sf /usr/local/lib/libsp_api.so.1 /usr/local/lib/libsp_api.so
 
This should create the necessary symlinks to the main library and place them in the library directory.

The shared library can now be linked in with g++ by
    g++ sources -o output_exe -Wl, -rpath /usr/local/lib -lsp_api


