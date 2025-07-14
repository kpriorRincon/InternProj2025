Signal Hound BB60D/C/A API for 64-bit Linux systems

The API was developed and tested on Ubuntu 18.04 LTS and CentOS 7
Installation instructions may reflect this.

-- Description --
The BB60 API is now available for 64-bit Linux. You can use the API to control both the BB60D/BB60C/BB60A Signal Hound spectrum analyzers. 
The API functions identical to the Windows version, and as such, share examples and documentation. 

-- Compilation Notes -- 
The API is compiled as a shared object library.
The API depends on libusb-1.0 and the FTDI USB drivers. Installation instructions can be found below for both libraries.

-- Libusb 1.0 requirement -- 
You will need libusb-1.0 installed in the system path. 
Determine if libusb-1.0 is installed on your system with
  'locate libusb-1.0.so'
You can install libusb-1.0 with
  'sudo apt-get install libusb-1.0-0' 
or
  download and install from www.libusb.org

-- FTDI D2XX library requirement --
You will need the FTDI D2XX USB library (libftd2xx.so) in your system path.
You can find the latest library at
www.ftdichip.com/Drivers/D2XX.htm and find the 64-bit Linux driver download on the download table 
or
You can find a copy of the library in the lib folder shipped with this SDK. 
The version of the distributed library is the latest version at the time of development for 64-bit systems. 

-- Device Permissions --
To run an application utilizing libusb, you need to be have
root permissions. You can either run your application as root or change permissions for the device.

-- Changing Device Permissions --
If you do not want to run your application with root permissions, you will need to change permissions for the Signal Hound BB60 devices. 
You will need to place the sh_usb.rules file in the /etc/udev/rules.d/ directory. 
Once you have done this, you will need to unplug and plug in the device for the rules to take effect. 
You should then be able to interface the device without root permissions.

-- Installation --
The BB60 API is a 64 bit shared library. The library can be found in the lib/ folder.
The best way to develop to the bb_api is to place the included libraries in the /usr/local/lib directory. Steps to perform this are below.

To install the shared library on your system from the lib folder type

    sudo cp libbb_api.* /usr/local/lib
    sudo ldconfig -v -n /usr/local/lib
    sudo ln -sf /usr/local/lib/libbb_api.so.5 /usr/local/lib/libbb_api.so

This should create the necessary symlinks to the main library and place them in the library directory.

This same process can be performed for the ftdi library if you wish to install the one distributed.

The shared library can now be linked in with g++ by
    g++ sources -o output_exe -Wl,-rpath /usr/local/lib -lbb_api

Contact: support@signalhound.com



