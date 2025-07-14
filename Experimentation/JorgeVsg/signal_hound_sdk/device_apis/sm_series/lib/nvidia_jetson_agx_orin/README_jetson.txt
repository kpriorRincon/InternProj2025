Signal Hound SM200/435 Programming API for the Nvidia Jetson Orin AGX

Email/Support: support@signalhound.com

-- Documentation -- 
https://signalhound.com/sigdownloads/SDK/online_docs/sm_api/index.html

-- Limitations --
The follow limitations apply to the ARM build of the API,
- The API is limited to the sweep and I/Q streaming modes only.
- In sweep mode, RBW and VBW is limited to 12.5Hz.
- In the non-fast sweep mode, the sweep speed will be reduced, fast-sweep mode will be minimally impacted.
- In I/Q streaming mode, decimation is limited to values of 1, 2, and 4.
- In I/Q streaming mode, the software filter cannot be disabled. This means there can be
  spurious and aliased signals in the rejection regions. The passband will not be affected. 
  If spurious signals in the reject bands are not acceptable, we recommend
  running an additional post acquisition FIR filter.

-- Compilation Notes -- 
The API is compiled on the Nvidia Jetson Orin AGX dev kit
- Ubuntu 20.04 using g++ 9.4.0

-- Libusb 1.0 requirement -- 
The API depends on the libusb-1.0 USB drivers.
You will need libusb-1.0 installed in the system path. 
Determine if libusb-1.0 is installed on your system with
  'locate libusb-1.0.so'
You can install libusb-1.0 with
  'sudo apt-get install libusb-1.0-0' 
or
  download and install from www.libusb.org

-- Device Permissions --
To run an application utilizing libusb, you need to be have root permissions. 
You can either run your application as root or change permissions for the device.

-- Changing Device Permissions --
If you do not want to run your application with root permissions, 
you will need to change permissions for the SM200/435 device. 
You will need to place the sh_usb.rules file in the /etc/udev/rules.d/ directory. 
Once you have done this, you will need to unplug and plug in the device for the rules to take effect. 
You should then be able to interface the device without root permissions.

-- Installation --
This is a 64 bit shared library. The best way to use the sp_api is to place the included libraries in the /usr/local/lib directory. 

Steps to perform this are below.

To install the shared library on your system from the lib folder, type

    sudo cp libsm_api.* /usr/local/lib
    sudo ldconfig -v -n /usr/local/lib
    sudo ln -sf /usr/local/lib/libsm_api.so.1 /usr/local/lib/libsm_api.so
 
This should create the necessary symlinks to the main library and place them in the library directory.

The shared library can now be linked in with g++ by
    g++ sources -o output_exe -Wl, -rpath /usr/local/lib -lsm_api


