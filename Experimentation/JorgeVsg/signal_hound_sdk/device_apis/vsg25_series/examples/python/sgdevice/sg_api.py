# -*- coding: utf-8 -*-

# Copyright (c) 2022 Signal Hound
# For licensing information, please see the API license in the software_licenses folder

from ctypes import *
import numpy
from sys import exit

sglib = CDLL("sgdevice/sg_api.dll")


# ---------------------------------- CONSTANTS -----------------------------------

SG_FALSE = 0
SG_TRUE = 1

SG_MAX_DEVICES = 16

SG_MIN_FREQUENCY = 80.0e6
SG_MAX_FREQUENCY = 2.55e9
SG_MIN_OUTPUT_POWER = -50.0
SG_MAX_OUTPUT_POWER = +13.0

SG_MIN_AM_FREQ = 30.0
SG_MAX_AM_FREQ = 50.0e6

SG_MIN_SYMBOL_RATE = 53.334e3
SG_MAX_SYMBOL_RATE = 180.0e6

SG_MODE_CW = 0
SG_MODE_AM = 1
SG_MODE_FM = 2
SG_MODE_Pulse = 3
SG_MODE_PM = 4
SG_MODE_STEP_SWEEP = 5
SG_MODE_LIST_SWEEP = 6
SG_MODE_NOISE = 7 # N/A
SG_CUSTOM_IQ = 8

SG_PARABOLIC = 0
SG_RANDOM = 1
SG_RANDOM_FIXED_SEED = 2
SG_ZERO_PHASE = 3

SG_SHAPE_SINE = 0
SG_SHAPE_TRIANGLE = 1
SG_SHAPE_SQUARE = 2
SG_SHAPE_RAMP = 3

SG_RAISED_COSINE = 0
SG_ROOT_RAISED_COSINE = 1
SG_GAUSSIAN = 2
SG_NONE = 3

SG_MOD_BPSK = 0
SG_MOD_DBPSK = 1
SG_MOD_QPSK = 2
SG_MOD_DQPSK = 3
SG_MOD_OQPSK = 4
SG_MOD_PI4DQPSK = 5
SG_MOD_8PSK = 6
SG_MOD_D8PSK = 7
SG_MOD_16PSK = 8
SG_MOD_16QAM = 9
SG_MOD_64QAM = 10
SG_MOD_256QAM = 11
SG_MOD_8VSB = 20 # Experimental


# --------------------------------- BINDINGS ----------------------------------

sgGetDeviceList = sglib.sgGetDeviceList
sgGetDeviceList.argtypes = [
    numpy.ctypeslib.ndpointer(c_int, ndim=1, flags='C'),
    POINTER(c_int)
]

sgOpenDevice = sglib.sgOpenDevice
sgOpenDeviceBySerial = sglib.sgOpenDeviceBySerial
sgCloseDevice = sglib.sgCloseDevice
sgGetSerialNumber = sglib.sgGetSerialNumber
sgSetFrequencyAmplitude = sglib.sgSetFrequencyAmplitude
sgRFOff = sglib.sgRFOff
sgSetCW = sglib.sgSetCW
sgSetAM = sglib.sgSetAM
sgSetFM = sglib.sgSetFM
sgSetPulse = sglib.sgSetPulse
sgSetSweep = sglib.sgSetSweep
sgSetMultitone = sglib.sgSetMultitone

sgSetASK = sglib.sgSetASK
sgSetASK.argtypes = [
    c_int, c_double, c_int, c_double, c_double,
    numpy.ctypeslib.ndpointer(c_int, ndim=1, flags='C'),
    c_int
]

sgSetFSK = sglib.sgSetFSK
sgSetFSK.argtypes = [
    c_int, c_double, c_int, c_double, c_double,
    numpy.ctypeslib.ndpointer(c_int, ndim=1, flags='C'),
    c_int
]

sgSetPSK = sglib.sgSetPSK
sgSetPSK.argtypes = [
    c_int, c_double, c_int, c_int, c_double,
    numpy.ctypeslib.ndpointer(c_int, ndim=1, flags='C'),
    c_int
]

sgSetVSB = sglib.sgSetVSB
sgSetVSB.argtypes = [
    c_int, c_double,
    numpy.ctypeslib.ndpointer(c_int, ndim=1, flags='C'),
    c_int
]

sgSetCustomIQ = sglib.sgSetCustomIQ
sgSetCustomIQ.argtypes = [
    c_int, c_double,
    numpy.ctypeslib.ndpointer(numpy.double, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(numpy.double, ndim=1, flags='C'),
    c_int, c_int
]

sgQueryPulse = sglib.sgQueryPulse
sgQuerySymbolClockRate = sglib.sgQuerySymbolClockRate
sgQueryClockError = sglib.sgQueryClockError
sgSetIQNullValue = sglib.sgSetIQNullValue

sgGetStatusString = sglib.sgGetStatusString
sgGetStatusString.restype = c_char_p

sgGetAPIVersion = sglib.sgGetAPIVersion
sgGetAPIVersion.restype = c_char_p


# ---------------------------------- Utility ----------------------------------

def error_check(func):
    def print_status_if_error(*args, **kwargs):
        return_vars = func(*args, **kwargs)
        if "status" not in return_vars.keys():
            return return_vars
        status = return_vars["status"]
        if status != 0:
            print (f"{'Error' if status < 0 else 'Warning'} {status}: {sg_get_status_string(status)} in {func.__name__}()")
        if status < 0:
            exit()
        return return_vars
    return print_status_if_error


# --------------------------------- Functions ---------------------------------

@error_check
def sg_get_device_list():
    serials = numpy.zeros(SG_MAX_DEVICES).astype(c_int)
    device_count = c_int(-1)

    status = sgGetDeviceList(serials, byref(device_count))

    return {
        "status": status,
        "serials": serials,
        "device_count": device_count.value
    }

@error_check
def sg_open_device():
    device = c_int(-1)

    status = sgOpenDevice(byref(device))

    return {
        "status": status,
        "handle": device.value
    }

@error_check
def sg_open_device_by_serial(serial_number):
    device = c_int(-1)

    status = sgOpenDeviceBySerial(byref(device), serial_number)

    return {
        "status": status,
        "handle": device.value
    }

@error_check
def sg_close_device(device):
    return {
        "status": sgCloseDevice(device)
    }

@error_check
def sg_get_serial_number(device):
    serial = c_int(-1)

    status = sgGetSerialNumber(device, byref(serial))
    return {
        "status": status,
        "serial": serial.value
    }

@error_check
def sg_set_frequency_amplitude(device, frequency, amplitude):
    return {
        "status": sgSetFrequencyAmplitude(device, c_double(frequency), c_double(amplitude))
    }

@error_check
def sg_rf_off(device):
    return {
        "status": sgRFOff(device)
    }

@error_check
def sg_set_cw(device):
    return {
        "status": sgSetCW(device)
    }

@error_check
def sg_set_am(device, frequency, depth, shape):
    return {
        "status": sgSetAM(device, c_double(frequency), c_double(depth), shape)
    }

@error_check
def sg_set_fm(device, frequency, deviation, shape):
    return {
        "status": sgSetFM(device, c_double(frequency), c_double(deviation), shape)
    }

@error_check
def sg_set_pulse(device, period, width):
    return {
        "status": sgSetPulse(device, c_double(period), c_double(width))
    }

@error_check
def sg_set_sweep(device, time, span):
    return {
        "status": sgSetSweep(device, c_double(time), c_double(span))
    }

@error_check
def sg_set_multitone(device, count, spacing, notchWidth, phase):
    return {
        "status": sgSetMultitone(device, count, c_double(spacing), c_double(notchWidth), phase)
    }

@error_check
def sg_set_ask(device, symbolRate, filterType, filterAlpha, depth, symbols, symbolCount):
    return {
        "status": sgSetASK(device, c_double(symbolRate), filterType, c_double(filterAlpha), c_double(depth), symbols, symbolCount)
    }

@error_check
def sg_set_fsk(device, symbolRate, filterType, filterAlpha, modulationIndex, symbols, symbolCount):
    return {
        "status": sgSetFSK(device, c_double(symbolRate), filterType, c_double(filterAlpha), c_double(modulationIndex), symbols, symbolCount)
    }

@error_check
def sg_set_psk(device, symbolRate, modType, filterType, filterAlpha, symbols, symbolCount):
    return {
        "status": sgSetPSK(device, c_double(symbolRate), modType, filterType, c_double(filterAlpha), symbols, symbolCount)
    }

@error_check
def sg_set_vsb(device, symbolRate, symbols, symbolCount):
    return {
        "status": sgSetVSB(device, c_double(symbolRate), symbols, symbolCount)
    }

@error_check
def sg_set_custom_iq(device, clockRate, IVals, QVals, length, period):
    return {
        "status": sgSetCustomIQ(device, c_double(clockRate), IVals, QVals, length, period)
    }

@error_check
def sg_query_pulse(device):
    period = c_double(-1)
    width = c_double(-1)

    status = sgQueryPulse(device, byref(period), byref(width))

    return {
        "status": status,
        "period": period.value,
        "width": width.value
    }

@error_check
def sg_query_symbol_clock_rate(device):
    clock = c_double(-1)

    status = sgQuerySymbolClockRate(device, byref(clock))

    return {
        "status": status,
        "clock": clock.value
    }

@error_check
def sg_query_clock_error(device):
    error = c_double(-1)

    status = sgQueryClockError(device, byref(error))

    return {
        "status": status,
        "error": error.value
    }

@error_check
def sg_set_iq_null_value(device, Icount, Qcount):
    return {
        "status": sgSetIQNullValue(device, Icount, Qcount)
    }

@error_check
def sg_get_status_string(status):
    return {
        "status_string": sgGetStatusString(status)
    }

@error_check
def sg_get_api_version():
    return {
        "api_version": sgGetAPIVersion()
    }
