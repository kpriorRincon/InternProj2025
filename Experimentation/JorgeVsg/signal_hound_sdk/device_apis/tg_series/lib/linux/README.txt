Signal Hound TG44/124A API for 64-bit Linux systems

The API was developed and tested on Ubuntu 14.04 LTS
Installation instructions may reflect this.

Email/Support: roger@signalhound.com

Release Notes: 
2/8/2017 : API Version 1.1.0 compiled for 64-bit Linux.

-- Description --
The TG44/124A API is now available for 64-bit Linux. You can use the API to control both the TG44A and TG124A Signal Hound tracking generators. The API will function identical to the Windows version, and as such, share examples and documentation.
You can find the latest manuals at
https://signalhound.com/support/product-downloads/tg44a-tg124a-downloads/

-- Compilation Notes -- 
The API is compiled on Ubuntu 14.04 using g++ 4.8.2 as a shared object library.
The API depends on libusb-1.0 and ftdi USB drivers. Installation instructions can be found below for both libraries.

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
You can find a copy of the library in the lib folder shipped with this SDK. The version of the distributed library is the latest version at the time of development for 64-bit systems. 

-- Device Permissions --
To run an application utilizing libusb, you need to be have
root permissions. You can either run your application as root or change permissions for the device.

-- Changing Device Permissions --
If you do not want to run your application with root permissions, you will need to change permissions for the Signal Hound TG44/124 devices. You will need to place the tg.rules file in the /etc/udev/rules.d/ directory. Once you have done this, you will need to restart the computer for the rules to take effect. You should then be able to interface the device without root permissions.

-- Installation --
The TG44/124 API is a 64 bit shared library. The library can be found in the lib/ folder.
The best way to develop to the tg_api is to place the included libraries in the /usr/local/lib directory. Steps to perform this are below.

To install the shared library on your system from the lib folder type

    sudo cp libtg_api.* /usr/local/lib
    sudo ldconfig -v -n /usr/local/lib
    sudo ln -sf /usr/local/lib/libtg_api.so.1 /usr/local/lib/libtg_api.so
 
This should create the necessary symlinks to the main library and place them in the library directory.

The shared library can now be linked in with g++ by
    g++ sources -o output_exe -Wl,-rpath=/usr/local/lib -ltg_api

This same process can be performed for the ftdi library if you wish to install the one distributed.

To prevent some kernel modules from loading automatically you will want to copy the blacklist file 
  'sudo cp hound-blacklist.conf /etc/modprobe.d/hound-blacklist.conf'

You will need to reboot for all changes to take effect.
  

