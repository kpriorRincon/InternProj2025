# -*- coding: utf-8 -*-

# Copyright (c) 2024 Signal Hound
# For licensing information, please see the API license in the software_licenses folder

from ctypes import CDLL
from ctypes import POINTER
from ctypes import byref
from ctypes import c_int
from ctypes import c_double
from ctypes import c_uint
from ctypes import c_char_p

from enum import Enum

from sys import exit

pnlib = CDLL("pndevice/pn_api.dll")


# ---------------------------------- Constants ---------------------------------

# Device vendor identification number.
PN400_VID = 0x2817
# Device product identification number.
PN400_PID = 0x0104

PN400_FW = 0x1

# Maximum number of devices that can be interfaced in the API.
PN_MAX_DEVICES = 8

# Minimum voltage for the VCO tune port.
PN_VTUNE_MIN = -1.0
# Maximum voltage for the VCO tune port.
PN_VTUNE_MAX = 28.0

# Minimum voltage for the VCO supply port.
PN_VSUPPLY_MIN = 0.5
# Maximum voltage for the VCO supply port.
PN_VSUPPLY_MAX = 15.25

# Maximum expected VCO supply port settling time in ms.
PN_VSUPPLY_SETTLING_TIME_MS = 6000

# Status code returned from all PN API functions.
class PnStatus(Enum):
    # Unsupported platform
    pnInvalidOSErr = -9
    # Invalid or missing correction data
    pnInvalidCorrectionsErr = -8
    # Invalid device handle
    pnInvalidDeviceErr = -7
    # Invalid device info
    pnInvalidDeviceInfoErr = -6
    # Invalid voltage
    pnInvalidVoltageErr = -5
    # Invalid serial port number
    pnInvalidSerialPortNumberErr = -4
    # One or more required pointer parameters are null.
    pnNullPtrErr = -3
    # Device disconnected
    pnConnectionLostErr = -2
    # Unable to open device
    pnDeviceNotFoundErr = -1

    # Function returned successfully
    pnNoError = 0


# --------------------------------- Bindings -----------------------------------

pnOpenDevice = pnlib.pnOpenDevice
pnCloseDevice = pnlib.pnCloseDevice
pnEnableOutput = pnlib.pnEnableOutput
pnSetVcc = pnlib.pnSetVcc
pnSetVtune = pnlib.pnSetVtune
pnTrigger = pnlib.pnTrigger
pnGetDiagnostics = pnlib.pnGetDiagnostics
pnGetSerialAndFirmware = pnlib.pnGetSerialAndFirmware
pnGetAPIVersion = pnlib.pnGetAPIVersion
pnGetAPIVersion.restype = c_char_p


# ---------------------------------- Utility ----------------------------------

def error_check(func):
    def print_status_if_error(*args, **kwargs):
        return_vars = func(*args, **kwargs)
        if "status" not in return_vars.keys():
            return return_vars
        status = return_vars["status"]
        if status != 0:
            print (f"{'Error' if status < 0 else 'Warning'} {status}: {status} in {func.__name__}()")
        if status < 0:
            exit()
        return return_vars
    return print_status_if_error

    def bool_to_int(b):
        return 1 if b is True else 0


# --------------------------------- Functions ---------------------------------

@error_check
def pn_open_device(serial_port):
    device = c_int(-1)

    status = pnOpenDevice(serial_port, byref(device))

    return {
        "status": status,

        "device": device.value
    }

@error_check
def pn_close_device(device):
    return {
        "status": pnCloseDevice(device)
    }

@error_check
def pn_enable_output(device, enabled):
    return {
        "status": pnEnableOutput(device, bool_to_int(enabled))
    }

@error_check
def pn_set_vcc(device, voltage):
    return {
        "status": pnSetVcc(device, c_double(voltage))
    }

@error_check
def pn_set_vtune(device, voltage):
    return {
        "status": pnSetVtune(device, c_double(voltage))
    }

@error_check
def pn_trigger(device):
    return {
        "status": pnTrigger(device)
    }

@error_check
def pn_get_diagnostics(device):
    usb_voltage = c_double(-1)
    dc_input_voltage = c_double(-1)
    vco_voltage = c_double(-1)
    vco_current = c_double(-1)

    status = pnGetDiagnostics(device,
                              byref(usb_voltage),
                              byref(dc_input_voltage),
                              byref(vco_voltage),
                              byref(vco_current))

    return {
        "status": status,

        "usb_voltage": usb_voltage.value,
        "dc_input_voltage": dc_input_voltage.value,
        "vco_voltage": vco_voltage.value,
        "vco_current": vco_current.value
    }

@error_check
def pn_get_serial_and_firmware(device):
    serial = c_uint(-1)
    firmware = c_uint(-1)

    status = pnGetSerialAndFirmware(device, byref(serial), byref(firmware))

    return {
        "status": status,

        "serial": serial.value,
        "firmware": firmware.value
    }

@error_check
def pn_get_api_version():
    return {
        "status": pnGetAPIVersion()
    }
