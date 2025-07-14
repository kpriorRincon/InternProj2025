Contact Information: support@signalhound.com

Prerequisites:
Python 3 is required to use the Python interface functions. The API has been
tested with Python 3.6.1 on Windows.

The VSG25A software must be fully installed before using the Python
interface functions. Additionally, ensure the VSG25A is stable and
functioning in the software before using the Python interface.

The NumPy C-Types Foreign Function Interface package is required to use the Python interface
functions, due to the need for fast array processing. The easiest way is to
install the SciPy stack, available here: https://www.scipy.org/install.html.
Alternatively, the necessary individual packages could be installed.

For our tests we used the 64-bit version of Anaconda for Windows.

Purpose:
This project enables the user to generate signals from the VSG25A through a
number of convenience functions written in Python. These functions serve as
a binding to our SG API, and perform the majority of C DLL interfacing
necessary to communicate with the generator.

Setup:
Place the sg_api.py and sg_api.dll files into the sgdevice/ folder.

To call the functions from outside the sgdevice/ directory you may need to add the
sgdevice folder to the Python search path and to the system path. This can be done
by editing PATH and PYTHONPATH in: Control Panel > System and Security > System
> Advanced system settings > Advanced > Environment Variables > System variables.

To run the example scripts, navigate to the folder containing the example
.py files. Each example file is standalone and provides example code for calling the
provided Python functions for the VSG25A.

Usage:
The functions under the "Functions" heading are callable from external scripts.
They are functionally equivalent to their C counterparts, except memory management
is handled by the API instead of the user.
