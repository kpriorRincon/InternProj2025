Signal Hound SP145 Programming API for 64-bit Linux systems

Email/Support: support@signalhound.com

-- Documentation -- 
https://signalhound.com/sigdownloads/SDK/online_docs/sp_api/index.html

-- Compilation Notes -- 
The API is compiled on
- Ubuntu 18.04 using g++ 7.3.0
- CentOS 7 using g++
- RedHat 7/8 using g++

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
you will need to change permissions for the SP145 device. 
You will need to place the sh_usb.rules file in the /etc/udev/rules.d/ directory. 
Once you have done this, you will need to unplug and plug in the device for the rules to take effect. 
You should then be able to interface the device without root permissions.

-- Installation --
This is a 64 bit shared library. The best way to use the sp_api is to place the included libraries in the /usr/local/lib directory. 

Steps to perform this are below.

To install the shared library on your system from the lib folder, type

    sudo cp libsp_api.* /usr/local/lib
    sudo ldconfig -v -n /usr/local/lib
    sudo ln -sf /usr/local/lib/libsp_api.so.1 /usr/local/lib/libsp_api.so
 
This should create the necessary symlinks to the main library and place them in the library directory.

The shared library can now be linked in with g++ by
    g++ sources -o output_exe -Wl, -rpath /usr/local/lib -lsp_api


