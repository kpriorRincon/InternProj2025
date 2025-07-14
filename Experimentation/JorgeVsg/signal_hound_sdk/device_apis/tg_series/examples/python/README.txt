Contact Information: support@signalhound.com

Prerequisites:
Python 3 is required to use the Python interface functions. The API has been
tested with Python 3.6.1 on Windows.

The Spike software must be fully installed before using the Python
interface functions. Additionally, ensure the TG44A/124A is stable and
functioning in Spike before using the Python interface.

Purpose:
This project enables the user to configure the frequency and amplitude of
a generated signal through a number of convenience functions
written in Python. These functions serve as a binding to our TG API,
and perform the majority of C DLL interfacing necessary to communicate
with the receiver.

Setup:
Place the tg_api.py and tg_api.dll files into the tgdevice/ folder.

To call the functions from outside the tgdevice/ directory you may need to add the
tgdevice folder to the Python search path and to the system path. This can be done
by editing PATH and PYTHONPATH in: Control Panel > System and Security > System
> Advanced system settings > Advanced > Environment Variables > System variables.

To run the example scripts, navigate to the folder containing the example
.py files. Each example file is standalone and provides example code for calling the
provided Python functions for the TG44A/124A.

To use on Linux, point the binding to the shared object instead of the DLL.
To do this, edit the line in tg_api.py from

    tglib = CDLL("tgdevice/tg_api.dll")
        to
    tglib = CDLL("tgdevice/tg_api.so").

Usage:
The functions under the "Functions" heading are callable from external scripts.
They are functionally equivalent to their C counterparts, except memory management
is handled by the API instead of the user.
