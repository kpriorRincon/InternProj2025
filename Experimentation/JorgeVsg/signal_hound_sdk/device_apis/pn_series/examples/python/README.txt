# Python Programming Language Compatibility

This project enables the user to operate the PN400 Phase Noise and VCO Tester through a number of convenience functions written in Python. These functions serve as a mostly one-to-one binding to the native API, and perform the majority of C shared library interfacing necessary to communicate with the receiver.

## Prerequisites

Python 3 is required to use the Python interface functions. The API has been tested with Python 3.6.1 on Windows.

The Spike software must be fully installed before using the Python interface functions. Additionally, ensure the PN400 device is stable and functioning in Spike before using the Python interface.

For our tests we used the 64-bit version of Anaconda for Windows.

## Setup

### Windows

Place pn_api.dll into the pndevice/ folder, along with pn_api.py.

To call the functions from outside the pndevice/ directory you may need to add the pndevice folder to the Python search path and to the system path. This can be done by editing PATH and PYTHONPATH in: Control Panel > System and Security > System > Advanced system settings > Advanced > Environment Variables > System variables.

### Linux

Place pn_api.so into the pndevice/ folder, along with pn_api.py.

Edit the line in pn_api.py from

    pnlib = CDLL("pndevice/pn_api.dll")
        to
    pnlib = CDLL("pndevice/pn_api.so")

## Usage

The functions under the "Public" heading are callable from external scripts. They are functionally equivalent to their C counterparts, except memory management is handled by the API instead of the user.

To run the example scripts, navigate to the folder containing the example .py files. Each example file is standalone and provides example code for calling the provided Python functions for the PN400.

## Support

Contact Information: support@signalhound.com
