# -*- coding: utf-8 -*-

# Copyright (c) 2025 Signal Hound
# For licensing information, please see the API license in the software_licenses folder

from ctypes import CDLL
from ctypes import POINTER
from ctypes import byref
from ctypes import Structure
from ctypes import c_int
from ctypes import c_longlong
from ctypes import c_float
from ctypes import c_double
from ctypes import c_uint
from ctypes import c_ulonglong
from ctypes import c_ubyte
from ctypes import c_char
from ctypes import c_char_p
from ctypes import c_bool

from enum import Enum

import numpy
from sys import exit

splib = CDLL("spdevice/sp_api.dll")


# ---------------------------------- Constants ---------------------------------

SP_INVALID_HANDLE = -1

# Used for boolean true when integer parameters are being used. Also see #SpBool.
SP_TRUE = 1
# Used for boolean false when integer parameters are being used. Also see #SpBool.
SP_FALSE = 0

# Max number of devices that can be interfaced in the API.
SP_MAX_DEVICES = 9

# Maximum reference level in dBm
SP_MAX_REF_LEVEL = 20.0
# Tells the API to automatically choose attenuation based on reference level.
SP_AUTO_ATTEN = -1
# Valid atten values [0,6] or -1 for auto
SP_MAX_ATTEN = 6

# Min frequency for sweeps, and min center frequency for I/Q measurements.
SP_MIN_FREQ = 9.0e3
# Max frequency for sweeps, and max center frequency for I/Q measurements.
SP_MAX_FREQ = 15.0e9

# Min sweep time in seconds. See #spSetSweepCoupling.
SP_MIN_SWEEP_TIME = 1.0e-6
# Max sweep time in seconds. See #spSetSweepCoupling.
SP_MAX_SWEEP_TIME = 100.0

# Min span for device configured in real-time measurement mode
SP_REAL_TIME_MIN_SPAN = 200.0e3
# Max span for device configured in real-time measurement mode
SP_REAL_TIME_MAX_SPAN = 40.0e6
# Min RBW for device configured in real-time measurement mode
SP_REAL_TIME_MIN_RBW = 2.0e3
# Max RBW for device configured in real-time measurement mode
SP_REAL_TIME_MAX_RBW = 1.0e6

# Max decimation for I/Q streaming.
SP_MAX_IQ_DECIMATION = 8192

# Maximum number of sweeps that can be queued up. Valid sweep indices between [0,15]
SP_MAX_SWEEP_QUEUE_SZ = 16

# Status code returned from all SP API functions.
class SpStatus(Enum): # TODO: Assign values
    # Unable to open device
    spDeviceNotFoundErr = -100
    # Unable to allocate resources needed to configure the measurement mode
    spAllocationErr = -101
    spTransferErr = -102
    # Only can connect up to SP_MAX_DEVICES receivers
    spMaxDevicesConnectedErr = -103
    # Required parameter found to have invalid value
    spInvalidParameterErr = -104
    # User specified invalid device index
    spInvalidDeviceErr = -105
    # One or more required pointer parameters were null
    spNullPtrErr = -106
    # Device disconnected, likely USB error detected
    spConnectionLostErr = -107

     # Attempting to perform an operation that cannot currently be performed.
     # Often the result of trying to do something while the device is currently
     # making measurements or not in an idle state.

    spInvalidConfigurationErr = -108
    spSweepAlreadyActiveErr = -109
    spFileIOErr = -110
    # If the core FX3 program fails to run
    spFx3RunErr = -111
    # Boot error
    spBootErr = -112
    # Invalid or already active sweep position
    spInvalidSweepPositionErr = -113
    spInvalidCalDataErr = -114

    # Function returned successfully
    spNoError = 0

    # One or more of the provided settings were adjusted
    spSettingClamped = 1
    # Temperature drift occured, measurements uncalibrated, reconfigure the device
    spTempDrift = 2
    # Measurement includes data which caused an ADC overload (clipping/compression)
    spADCOverflow = 3
    # Measurement is uncalibrated, overrides ADC overflow
    spUncalData = 4
    # Returned when the API was unable to keep up with the necessary processing
    spCPULimited = 5
    # Calibration data potentially corrupt
    spInvalidCalData = 6

# Device type
class SpDeviceType(Enum):
    # SP145A
    spDeviceTypeSP145A = 0

# Boolean type. Used in public facing functions instead of `bool` to improve
# API use from different programming languages.
class SpBool(Enum):
    # False
    spFalse = 0
    # True
    spTrue = 1

# Specifies device power state. See @ref powerStates for more information.
class SpPowerState(Enum):
    # On
    spPowerStateOn = 0
    # Standby
    spPowerStateStandby = 1

# Measurement mode
class SpMode(Enum):
    # Idle, no measurement
    spModeIdle = 0
    # Swept spectrum analysis
    spModeSweeping = 1
    # Real-time spectrum analysis
    spModeRealTime = 2
    # I/Q streaming
    spModeIQStreaming = 3
    # I/Q sweep list / frequency hopping
    spModeIQSweepList = 4
    # Audio demod
    spModeAudio = 5

# Detector used for sweep and real-time spectrum analysis.
class SpDetector(Enum):
    # Average
    spDetectorAverage = 0
    # Min/Max
    spDetectorMinMax = 1

# Specifies units of sweep and real-time spectrum analysis measurements.
class SpScale(Enum):
    # dBm
    spScaleLog = 0
    # mV
    spScaleLin = 1
    # Log scale, no corrections
    spScaleFullScale = 2

# Specifies units in which VBW processing occurs.
class SpVideoUnits(Enum):
    # dBm
    spVideoLog = 0
    # Linear voltage
    spVideoVoltage = 1
    # Linear power
    spVideoPower = 2
    # No VBW processing
    spVideoSample = 3

# Specifies the window used for sweep and real-time analysis.
class SpWindowType(Enum):
    # SRS flattop
    spWindowFlatTop = 0
    # Nutall
    spWindowNutall = 1
    # Blackman
    spWindowBlackman = 2
    # Hamming
    spWindowHamming = 3
    # Gaussian 6dB BW window for EMC measurements and CISPR compatibility
    spWindowGaussian6dB = 4
    # Rectangular (no) window
    spWindowRect = 5

# Specifies a data type for data returned from the API
class SpDataType(Enum):
    # 32-bit complex floats
    spDataType32fc = 0
    # 16-bit complex shorts
    spDataType16sc = 1

# Trigger edge for video and external triggers.
class SpTriggerEdge(Enum):
    # Rising edge
    spTriggerEdgeRising = 0
    # Falling edge
    spTriggerEdgeFalling = 1

# Internal GPS state
class SpGPSState(Enum):
    # GPS is not locked
    spGPSStateNotPresent = 0
    # GPS is locked, NMEA data is valid, but the timebase is not being disciplined by the GPS
    spGPSStateLocked = 1
    # GPS is locked, NMEA data is valid, timebase is being disciplined by the GPS
    spGPSStateDisciplined = 2

# Audio demodulation type.
class SpAudioType(Enum):
    # AM
    spAudioTypeAM = 0
    # FM
    spAudioTypeFM = 1
    # Upper side band
    spAudioTypeUSB = 2
    # Lower side band
    spAudioTypeLSB = 3
    # CW
    spAudioTypeCW = 4


# --------------------------------- Bindings -----------------------------------

spGetDeviceList = splib.spGetDeviceList
spGetDeviceList.argtypes = [
    numpy.ctypeslib.ndpointer(c_int, ndim=1, flags='C'),
    POINTER(c_int)
]

spOpenDevice = splib.spOpenDevice
spOpenDeviceBySerial = splib.spOpenDeviceBySerial
spCloseDevice = splib.spCloseDevice
spPresetDevice = splib.spPresetDevice

spSetPowerState = splib.spSetPowerState
spGetPowerState = splib.spGetPowerState

spGetSerialNumber = splib.spGetSerialNumber
spGetFirmwareVersion = splib.spGetFirmwareVersion
spGetDeviceDiagnostics = splib.spGetDeviceDiagnostics
spGetCalDate = splib.spGetCalDate

spSetReference = splib.spSetReference
spGetReference = splib.spGetReference

spSetGPSTimebaseUpdate = splib.spSetGPSTimebaseUpdate
spGetGPSTimebaseUpdate = splib.spGetGPSTimebaseUpdate
spGetGPSHoldoverInfo = splib.spGetGPSHoldoverInfo
spGetGPSState = splib.spGetGPSState

spSetRefLevel = splib.spSetRefLevel
spGetRefLevel = splib.spGetRefLevel

spSetAttenuator = splib.spSetAttenuator
spGetAttenuator = splib.spGetAttenuator

spSetSweepCenterSpan = splib.spSetSweepCenterSpan
spSetSweepStartStop = splib.spSetSweepStartStop
spSetSweepCoupling = splib.spSetSweepCoupling
spSetSweepDetector = splib.spSetSweepDetector
spSetSweepScale = splib.spSetSweepScale
spSetSweepWindow = splib.spSetSweepWindow

spSetRealTimeCenterSpan = splib.spSetRealTimeCenterSpan
spSetRealTimeRBW = splib.spSetRealTimeRBW
spSetRealTimeDetector = splib.spSetRealTimeDetector
spSetRealTimeScale = splib.spSetRealTimeScale
spSetRealTimeWindow = splib.spSetRealTimeWindow

spSetIQDataType = splib.spSetIQDataType
spSetIQCenterFreq = splib.spSetIQCenterFreq
spGetIQCenterFreq = splib.spGetIQCenterFreq
spSetIQSampleRate = splib.spSetIQSampleRate
spSetIQSoftwareFilter = splib.spSetIQSoftwareFilter
spSetIQBandwidth = splib.spSetIQBandwidth
spSetIQExtTriggerEdge = splib.spSetIQExtTriggerEdge
spSetIQTriggerSentinel = splib.spSetIQTriggerSentinel
spSetIQQueueSize = splib.spSetIQQueueSize

spSetIQSweepListDataType = splib.spSetIQSweepListDataType
spSetIQSweepListCorrected = splib.spSetIQSweepListCorrected
spSetIQSweepListSteps = splib.spSetIQSweepListSteps
spGetIQSweepListSteps = splib.spGetIQSweepListSteps
spSetIQSweepListFreq = splib.spSetIQSweepListFreq
spSetIQSweepListRef = splib.spSetIQSweepListRef
spSetIQSweepListAtten = splib.spSetIQSweepListAtten
spSetIQSweepListSampleCount = splib.spSetIQSweepListSampleCount

spSetAudioCenterFreq = splib.spSetAudioCenterFreq
spSetAudioType = splib.spSetAudioType
spSetAudioFilters = splib.spSetAudioFilters
spSetAudioFMDeemphasis = splib.spSetAudioFMDeemphasis

spConfigure = splib.spConfigure
spGetCurrentMode = splib.spGetCurrentMode
spAbort = splib.spAbort

spGetSweepParameters = splib.spGetSweepParameters
spGetRealTimeParameters = splib.spGetRealTimeParameters
spGetIQParameters = splib.spGetIQParameters
spGetIQCorrection = splib.spGetIQCorrection
spIQSweepListGetCorrections = splib.spIQSweepListGetCorrections
spIQSweepListGetCorrections.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C')
]

spGetSweep = splib.spGetSweep
spGetSweep.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    POINTER(c_longlong)
]

spGetRealTimeFrame = splib.spGetRealTimeFrame
spGetRealTimeFrame.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    POINTER(c_int),
    POINTER(c_longlong)
]

spGetIQ = splib.spGetIQ
spGetIQ.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.complex64, ndim=1, flags='C'),
    c_int,
    numpy.ctypeslib.ndpointer(c_double, ndim=1, flags='C'),
    c_int, POINTER(c_longlong), c_int, POINTER(c_int), POINTER(c_int)
]

spIQSweepListGetSweep = splib.spIQSweepListGetSweep
spIQSweepListGetSweep.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(c_longlong, ndim=1, flags='C')
]

spIQSweepListStartSweep = splib.spIQSweepListStartSweep
spIQSweepListStartSweep.argtypes = [
    c_int, c_int,
    numpy.ctypeslib.ndpointer(numpy.float32, ndim=1, flags='C'),
    numpy.ctypeslib.ndpointer(c_longlong, ndim=1, flags='C')
]

spIQSweepListFinishSweep = splib.spIQSweepListFinishSweep

spGetAudio = splib.spGetAudio
spGetAudio.argtypes = [
    c_int,
    numpy.ctypeslib.ndpointer(c_float, ndim=1, flags='C')
]

spGetGPSInfo = splib.spGetGPSInfo
spGetGPSInfo.argtypes = [
    c_int, c_int, POINTER(c_int), POINTER(c_longlong),
    POINTER(c_double), POINTER(c_double), POINTER(c_double),
    numpy.ctypeslib.ndpointer(c_char, ndim=1, flags='C'),
    POINTER(c_int)
]

spSetFanSetPoint = splib.spSetFanSetPoint
spGetFanSettings = splib.spGetFanSettings

spGetErrorString = splib.spGetErrorString
spGetErrorString.restype = c_char_p

spGetAPIVersion = splib.spGetAPIVersion
spGetAPIVersion.restype = c_char_p


# ---------------------------------- Utility ----------------------------------

def error_check(func):
    def print_status_if_error(*args, **kwargs):
        return_vars = func(*args, **kwargs)
        if "status" not in return_vars.keys():
            return return_vars
        status = return_vars["status"]
        if status != 0:
            print (f"{'Error' if status < 0 else 'Warning'} {status}: {sp_get_error_string(status)['error_string']} in {func.__name__}()")
        if status < 0:
            exit()
        return return_vars
    return print_status_if_error

def bool_to_int(b):
    return SP_TRUE if b is True else SP_FALSE

def int_to_bool(i):
    return True if i is SP_TRUE else False


# --------------------------------- Functions ---------------------------------

@error_check
def sp_get_device_list():
    serials = numpy.zeros(SP_MAX_DEVICES).astype(c_int)
    device_count = c_int(-1)

    status = spGetDeviceList(serials, byref(device_count))

    return {
        "status": status,

        "serials": serials,
        "device_count": device_count.value
    }

@error_check
def sp_open_device():
    device = c_int(-1)

    status = spOpenDevice(byref(device))

    return {
        "status": status,

        "device": device.value
    }

@error_check
def sp_open_device_by_serial(serial_number):
    device = c_int(-1)

    status = spOpenDeviceBySerial(byref(device), serial_number)

    return {
        "status": status,

        "device": device.value
    }

@error_check
def sp_close_device(device):
    return {
        "status": spCloseDevice(device)
    }

@error_check
def sp_preset_device(device):
    return {
        "status": spPresetDevice(device)
    }

@error_check
def sp_set_power_state(device, power_state):
    return {
        "status": spSetPowerState(device, power_state)
    }

@error_check
def sp_get_power_state(device):
    power_state = c_int(-1)

    status = spGetPowerState(device, byref(power_state))

    return {
        "status": status,

        "power_state": SpPowerState(power_state.value)
    }

@error_check
def sp_get_serial_number(device):
    serial_number = c_int(-1)

    status = spGetSerialNumber(device, byref(serial_number))

    return {
        "status": status,

        "serial_number": serial_number.value
    }

@error_check
def sp_get_firmware_version(device):
    major = c_int(-1)
    minor = c_int(-1)
    revision = c_int(-1)

    status = spGetFirmwareVersion(device, byref(major), byref(minor), byref(revision))

    return {
        "status": status,

        "major": major.value,
        "minor": minor.value,
        "revision": revision.value
    }

@error_check
def sp_get_device_diagnostics(device):
    voltage = c_float(-1)
    current = c_float(-1)
    temperature = c_float(-1)

    status = spGetDeviceDiagnostics(device, byref(voltage), byref(current), byref(temperature))

    return {
        "status": status,

        "voltage": voltage.value,
        "current": current.value,
        "temperature": temperature.value
    }

@error_check
def sp_get_cal_date(device):
    last_cal_date = c_ulong(0)

    status = spGetCalDate(device, byref(last_cal_date))

    return {
        "status": status,

        "last_cal_date": last_cal_date.value
    }

@error_check
def sp_set_reference(device, enabled):
    return {
        "status": spSetReference(device, bool_to_int(enabled))
    }

@error_check
def sp_get_reference(device):
    enabled = c_int(-1)

    status = spGetReference(device, byref(enabled))

    return {
        "status": status,

        "enabled": int_to_bool(enabled.value)
    }

@error_check
def sp_set_GPS_timebase_update(device, enabled):
    return {
        "status": spSetGPSTimebaseUpdate(device, bool_to_int(enabled))
    }

@error_check
def sp_get_GPS_timebase_update(device):
    enabled = c_int(-1)

    status = spGetGPSTimebaseUpdate(device, byref(enabled))

    return {
        "status": status,

        "enabled": int_to_bool(enabled.value)
    }

@error_check
def sp_get_GPS_holdover_info(device):
    using_GPS_holdover = c_int(-1)
    last_holdover_time = c_ulong(0)

    status = spGetGPSHoldoverInfo(device, byref(using_GPS_holdover),
                                  byref(last_holdover_time))

    return {
        "status": status,

        "using_GPS_holdover": int_to_bool(using_GPS_holdover.value),
        "last_holdover_time": last_holdover_time.value
    }

@error_check
def sp_get_GPS_state(device):
    GPS_state = c_int(-1)

    status = spGetGPSState(device, byref(GPS_state))

    return {
        "status": status,

        "GPS_state": SpGPSState(GPS_state.value)
    }

@error_check
def sp_set_ref_level(device, ref_level):
    return {
        "status": spSetRefLevel(device, c_double(ref_level))
    }

@error_check
def sp_get_ref_level(device):
    ref_level = c_double(-1)

    status = spGetRefLevel(device, byref(ref_level))

    return {
        "status": status,

        "ref_level": ref_level.value
    }

@error_check
def sp_set_attenuator(device, atten):
    return {
        "status": spSetAttenuator(device, atten)
    }

@error_check
def sp_get_attenuator(device):
    atten = c_int(-1)

    status = spGetAttenuator(device, byref(atten))

    return {
        "status": status,

        "atten": atten.value
    }

@error_check
def sp_set_sweep_center_span(device, center_freq_Hz, span_Hz):
    return {
        "status": spSetSweepCenterSpan(device,
                                       c_double(center_freq_Hz),
                                       c_double(span_Hz))
    }

@error_check
def sp_set_sweep_start_stop(device, start_freq_Hz, stop_freq_Hz):
    return {
        "status": spSetSweepStartStop(device,
                                      c_double(start_freq_Hz),
                                      c_double(stop_freq_Hz))
    }

@error_check
def sp_set_sweep_coupling(device, rbw, vbw, sweep_time):
    return {
        "status": spSetSweepCoupling(device,
                                     c_double(rbw), c_double(vbw),
                                     c_double(sweep_time))
    }

@error_check
def sp_set_sweep_detector(device, detector, video_units):
    return {
        "status": spSetSweepDetector(device, SpDetector(detector).value, SpVideoUnits(video_units).value)
    }

@error_check
def sp_set_sweep_scale(device, scale):
    return {
        "status": spSetSweepScale(device, SpScale(scale).value)
    }

@error_check
def sp_set_sweep_window(device, window):
    return {
        "status": spSetSweepWindow(device, SpWindowType(window).value)
    }

@error_check
def sp_set_real_time_center_span(device, center_freq_Hz, span_Hz):
    return {
        "status": spSetRealTimeCenterSpan(device, c_double(center_freq_Hz), c_double(span_Hz))
    }

@error_check
def sp_set_real_time_RBW(device, rbw):
    return {
        "status": spSetRealTimeRBW(device, c_double(rbw))
    }

@error_check
def sp_set_real_time_detector(device, detector):
    return {
        "status": spSetRealTimeDetector(device, SpDetector(detector).value)
    }

@error_check
def sp_set_real_time_scale(device, scale, frame_ref, frame_scale):
    return {
        "status": spSetRealTimeScale(device, SpScale(scale).value, c_double(frame_ref), c_double(frame_scale))
    }

@error_check
def sp_set_real_time_window(device, window):
    return {
        "status": spSetRealTimeWindow(device, SpWindowType(window).value)
    }

@error_check
def sp_set_IQ_data_type(device, data_type):
    return {
        "status": spSetIQDataType(device, SpDataType(data_type).value)
    }

@error_check
def sp_set_IQ_center_freq(device, center_freq_Hz):
    return {
        "status": spSetIQCenterFreq(device, c_double(center_freq_Hz))
    }

@error_check
def sp_get_IQ_center_freq(device):
    center_freq_Hz = c_double(-1)

    status = spGetIQCenterFreq(device, byref(center_freq_Hz))

    return {
        "status": status,

        "center_freq_Hz": center_freq_Hz.value
    }

@error_check
def sp_set_IQ_sample_rate(device, decimation):
    return {
        "status": spSetIQSampleRate(device, decimation)
    }

@error_check
def sp_set_IQ_software_filter(device, enabled):
    return {
        "status": spSetIQSoftwareFilter(device, bool_to_int(enabled))
    }

@error_check
def sp_set_IQ_bandwidth(device, bandwidth):
    return {
        "status": spSetIQBandwidth(device, c_double(bandwidth))
    }

@error_check
def sp_set_IQ_ext_trigger_edge(device, edge):
    return {
        "status": spSetIQExtTriggerEdge(device, SpTriggerEdge(edge).value)
    }

@error_check
def sp_set_IQ_trigger_sentinel(device, sentinel_value):
    return {
        "status": spSetIQTriggerSentinel(device, c_double(sentinel_value))
    }

@error_check
def sp_set_IQ_queue_size(device, ms):
    return {
        "status": spSetIQQueueSize(device, c_float(ms))
    }

@error_check
def sp_set_IQ_sweep_list_data_type(device, data_type):
    return {
        "status": spSetIQSweepListDataType(device, SpDataType(data_type).value)
    }

@error_check
def sp_set_IQ_sweep_list_corrected(device, corrected):
    return {
        "status": spSetIQSweepListCorrected(device, bool_to_int(corrected))
    }

@error_check
def sp_set_IQ_sweep_list_steps(device, steps):
    return {
        "status": spSetIQSweepListSteps(device, steps)
    }

@error_check
def sp_get_IQ_sweep_list_steps(device):
    steps = c_int(-1)

    status = spGetIQSweepListSteps(device, byref(steps))
    return {
        "status": status,

        "steps": steps.value
    }

@error_check
def sp_set_IQ_sweep_list_freq(device, step, freq):
    return {
        "status": spSetIQSweepListFreq(device, step, c_double(freq))
    }

@error_check
def sp_set_IQ_sweep_list_ref(device, step, level):
    return {
        "status": spSetIQSweepListRef(device, step, c_double(level))
    }

@error_check
def sp_set_IQ_sweep_list_atten(device, step, atten):
    return {
        "status": spSetIQSweepListAtten(device, step, atten)
    }

@error_check
def sp_set_IQ_sweep_list_sample_count(device, step, samples):
    return {
        "status": spSetIQSweepListSampleCount(device, step, c_ulong(samples))
    }

@error_check
def sp_set_audio_center_freq(device, center_freq_Hz):
    return {
        "status": spSetAudioCenterFreq(device, c_double(center_freq_Hz))
    }

@error_check
def sp_set_audio_type(device, audio_type):
    return {
        "status": spSetAudioType(device, SpAudioType(audio_type).value)
    }

@error_check
def sp_set_audio_filters(device, if_bandwidth, audio_lpf, audio_hpf):
    return {
        "status": spSetAudioFilters(device, c_double(if_bandwidth), c_double(audio_lpf), c_double(audio_hpf))
    }

@error_check
def sp_set_audio_FM_deemphasis(device, deemphasis):
    return {
        "status": spSetAudioFMDeemphasis(device,
                                         c_double(deemphasis))
    }

@error_check
def sp_configure(device, mode):
    return {
        "status": spConfigure(device,
                              SpMode(mode).value)
    }

@error_check
def sp_get_current_mode(device):
    mode = c_int(-1)

    status = spGetCurrentMode(device,
                              byref(mode))

    return {
        "status": status,

        "mode": SpMode(mode.value)
    }

@error_check
def sp_abort(device):
    return {
        "status": spAbort(device)
    }

@error_check
def sp_get_sweep_parameters(device):
    actual_RBW = c_double(-1)
    actual_VBW = c_double(-1)
    actual_start_freq = c_double(-1)
    bin_size = c_double(-1)
    sweep_size = c_int(-1)

    status = spGetSweepParameters(device,
                                  byref(actual_RBW),
                                  byref(actual_VBW),
                                  byref(actual_start_freq),
                                  byref(bin_size),
                                  byref(sweep_size))

    return {
        "status": status,

        "actual_RBW": actual_RBW.value,
        "actual_VBW": actual_VBW.value,
        "actual_start_freq": actual_start_freq.value,
        "bin_size": bin_size.value,
        "sweep_size": sweep_size.value
    }

@error_check
def sp_get_real_time_parameters(device):
    actual_RBW = c_double(-1)
    sweep_size = c_int(-1)
    actual_start_freq = c_double(-1)
    bin_size = c_double(-1)
    frame_width = c_int(-1)
    frame_height = c_int(-1)
    poi = c_double(-1)

    status = spGetRealTimeParameters(device,
                                     byref(actual_RBW),
                                     byref(sweep_size),
                                     byref(actual_start_freq),
                                     byref(bin_size),
                                     byref(frame_width),
                                     byref(frame_height),
                                     byref(poi))

    return {
        "status": status,

        "actual_RBW": actual_RBW.value,
        "sweep_size": sweep_size.value,
        "actual_start_freq": actual_start_freq.value,
        "bin_size": bin_size.value,
        "frame_width": frame_width.value,
        "frame_height": frame_height.value,
        "poi": poi.value
    }

@error_check
def sp_get_IQ_parameters(device):
    sample_rate = c_double(-1)
    bandwidth = c_double(-1)

    status = spGetIQParameters(device,
                               byref(sample_rate),
                               byref(bandwidth))

    return {
        "status": status,

        "sample_rate": sample_rate.value,
        "bandwidth": bandwidth.value
    }

@error_check
def sp_get_IQ_correction(device):
    scale = c_float(-1)

    status = spGetIQCorrection(device,
                               byref(scale))

    return {
        "status": status,

        "scale": scale.value
    }

@error_check
def sp_IQ_sweep_list_get_corrections(device):
    ret = sp_get_IQ_sweep_list_steps(device)
    if ret["status"] is not 0:
        return {
            "status": ret["status"]
        }
    steps = ret["steps"]

    corrections = numpy.zeros(steps).astype(numpy.float32)

    status = spIQSweepListGetCorrections(device,
                                         corrections)

    return {
        "status": status,

        "corrections": corrections
    }

@error_check
def sp_get_sweep(device):
    ret = sp_get_sweep_parameters(device)
    if ret["status"] is not 0:
        return {
            "status": ret["status"]
        }
    sweep_size = ret["sweep_size"]

    sweep_min = numpy.zeros(sweep_size).astype(numpy.float32)
    sweep_max = numpy.zeros(sweep_size).astype(numpy.float32)
    ns_since_epoch = c_longlong(-1)

    status = spGetSweep(device,
                        sweep_min,
                        sweep_max,
                        byref(ns_since_epoch))

    return {
        "status": status,

        "sweep_min": sweep_min,
        "sweep_max": sweep_max,
        "ns_since_epoch": ns_since_epoch.value
    }

@error_check
def sp_get_real_time_frame(device):
    ret = sp_get_real_time_parameters(device)
    if ret["status"] is not 0:
        return {
            "status": ret["status"]
        }
    sweep_size = ret["sweep_size"]
    frame_width = ret["frame_width"]
    frame_height = ret["frame_height"]

    color_frame = numpy.zeros(frame_width * frame_height).astype(numpy.float32)
    alpha_frame = numpy.zeros(frame_width * frame_height).astype(numpy.float32)
    sweep_min = numpy.zeros(sweep_size).astype(numpy.float32)
    sweep_max = numpy.zeros(sweep_size).astype(numpy.float32)
    frame_count = c_int(-1)
    ns_since_epoch = c_longlong(-1)

    status = spGetRealTimeFrame(device,
                                color_frame,
                                alpha_frame,
                                sweep_min,
                                sweep_max,
                                byref(frame_count),
                                byref(ns_since_epoch))
    return {
        "status": status,

        "color_frame": color_frame,
        "alpha_frame": alpha_frame,
        "sweep_min": sweep_min,
        "sweep_max": sweep_max,
        "frame_count": frame_count.value,
        "ns_since_epoch": ns_since_epoch.value
    }

@error_check
def sp_get_IQ(device, iq_buf_size, trigger_buf_size, purge):
    iq_buf = numpy.zeros(iq_buf_size).astype(numpy.complex64)
    triggers = numpy.zeros(trigger_buf_size).astype(c_double)
    ns_since_epoch = c_longlong(-1)
    sample_loss = c_int(-1)
    samples_remaining = c_int(-1)

    status = spGetIQ(device,
                     iq_buf,
                     iq_buf_size,
                     triggers,
                     trigger_buf_size,
                     byref(ns_since_epoch),
                     bool_to_int(purge),
                     byref(sample_loss),
                     byref(samples_remaining))

    return {
        "status": status,

        "iq_buf": iq_buf,
        "triggers": triggers,
        "ns_since_epoch": ns_since_epoch.value,
        "sample_loss": sample_loss.value,
        "samples_remaining": samples_remaining.value
    }

@error_check
def sp_IQ_sweep_list_get_sweep(device, samples, steps):
    dst = numpy.zeros(samples).astype(numpy.float32)
    timestamps = numpy.zeros(steps).astype(c_longlong)

    status = spIQSweepListGetSweep(device,
                                   dst,
                                   timestamps)

    return {
        "status": status,

        "dst": dst,
        "timestamps": timestamps
    }

@error_check
def sp_IQ_sweep_list_start_sweep(device, pos, samples, steps):
    dst = numpy.zeros(samples).astype(numpy.float32)
    timestamps = numpy.zeros(steps).astype(c_longlong)

    status = spIQSweepListStartSweep(device,
                                     pos,
                                     dst,
                                     timestamps)

    return {
        "status": status,

        "dst": dst,
        "timestamps": timestamps
    }

@error_check
def sp_IQ_sweep_list_finish_sweep(device, pos):
    return {
        "status": spIQSweepListFinishSweep(device,
                                           pos)
    }

@error_check
def sp_get_audio(device):
    audio = numpy.zeros(1000).astype(c_float)

    status = spGetAudio(device,
                        audio)

    return {
        "status": status,

        "audio": audio
    }

@error_check
def sp_get_GPS_info(device, refresh, nmea_len):
    updated = c_int(-1)
    sec_since_epoch = c_longlong(-1)
    latitude = c_double(-1)
    longitude = c_double(-1)
    altitude = c_double(-1)
    nmea = numpy.zeros(nmea_len.value).astype(c_char)

    status = spGetGPSInfo(device,
                          bool_to_int(refresh),
                          byref(updated),
                          byref(sec_since_epoch),
                          byref(latitude),
                          byref(longitude),
                          byref(altitude),
                          nmea,
                          byref(nmea_len))

    return {
        "status": status,

        "updated": int_to_bool(updated.value),
        "sec_since_epoch": sec_since_epoch.value,
        "latitude": latitude.value,
        "longitude": longitude.value,
        "altitude": altitude.value,
        "nmea": nmea,
        "nmea_len": nmea_len.value
    }

@error_check
def sp_set_fan_setpoint(device, temp):
    return {
        "status": spSetFanSetPoint(device,
                                    c_float(temp))
    }

@error_check
def sp_get_fan_settings(device):
    threshold = c_float(-1)
    voltage = c_float(-1)

    status = spGetFanSettings(device,
                              byref(threshold),
                              byref(voltage))

    return {
        "status": status,

        "threshold": threshold.value,
        "voltage": voltage.value
    }

def sp_get_error_string(status):
    return spGetErrorString(SpStatus(status).value)

def sp_get_API_version():
    return spGetAPIVersion()
