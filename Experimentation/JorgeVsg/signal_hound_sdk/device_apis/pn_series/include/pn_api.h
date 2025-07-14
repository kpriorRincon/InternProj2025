// Copyright (c).2024, Signal Hound, Inc.
// For licensing information, please see the API license in the software_licenses folder

/*!
 * \file pn_api.h
 * \brief API functions for the PN400 phase noise tester.
 *
 * This is the main file for user-accessible functions for controlling the
 * PN400 phase noise tester.
 *
 */

#ifndef PN_API_H
#define PN_API_H

#if defined(_WIN32) // Windows
#ifdef PN_EXPORTS
#define PN_API __declspec(dllexport)
#else
#define PN_API
#endif

 // bare minimum stdint typedef support
#if _MSC_VER < 1700 // For VS2010 or earlier
typedef signed char        int8_t;
typedef short              int16_t;
typedef int                int32_t;
typedef long long          int64_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
#else
#include <stdint.h>
#endif

#define PN_DEPRECATED(comment) __declspec(deprecated(comment))
#else // Linux
#include <stdint.h>
#define PN_API __attribute__((visibility("default")))

#if defined(__GNUC__)
#define PN_DEPRECATED(comment) __attribute__((deprecated))
#else
#define PN_DEPRECATED(comment) comment
#endif
#endif

/** Device vendor identification number. */
#define PN400_VID 0x2817
/** Device product identification number. */
#define PN400_PID 0x0104

#define PN400_FW 0x1

/** Maximum number of devices that can be interfaced in the API. */
#define PN_MAX_DEVICES 8

/** Minimum voltage for the VCO tune port. */
#define PN_VTUNE_MIN -1.0
/** Maximum voltage for the VCO tune port. */
#define PN_VTUNE_MAX 28.0

/** Minimum voltage for the VCO supply port. */
#define PN_VSUPPLY_MIN 0.5
/** Maximum voltage for the VCO supply port. */
#define PN_VSUPPLY_MAX 15.25

/** Maximum expected VCO supply port settling time in ms. */
#define PN_VSUPPLY_SETTLING_TIME 6000

typedef enum PnStatus {
    /** Unsupported platform */
    pnInvalidOSErr = -9,
    /** Invalid or missing correction data */
    pnInvalidCorrectionsErr = -8,
    /** Invalid device handle */
    pnInvalidDeviceErr = -7,
    /** Invalid device info */
    pnInvalidDeviceInfoErr = -6,
    /** Invalid voltage */
    pnInvalidVoltageErr = -5,
    /** Invalid serial port number */
    pnInvalidSerialPortNumberErr = -4,
    /** One or more required pointer parameters are null. */
    pnNullPtrErr = -3,
    /** Device disconnected */
    pnConnectionLostErr = -2,
    /** Unable to open device */
    pnDeviceNotFoundErr = -1,

    /** Function returned successfully */
    pnNoError = 0
} PnStatus;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Attempt to open a PN device on the specified serial port. If the device is
 * opened successfully, a handle to the function will be returned through the
 * _device_ pointer. This handle can then be used to refer to this device for
 * all future API calls.
 *
 * @param[in] serialPort The serial port on which to look for the device.
 * 
 * @param[out] device Returns handle that can be used to interface the device.
 * If this function returns an error, the handle will be invalid.
 *
 * @return
 */
PN_API PnStatus pnOpenDevice(int serialPort, int *device);

/**
 * This function should be called when you want to release the resources for a
 * device. All resources (memory, etc.) will be released, and the device will
 * become available again for use in the current process. The device handle
 * specified will no longer point to a valid device and the device must be
 * re-opened again to be used. This function should be called before the
 * process exits, but it is not strictly required.
 *
 * @param[in] device Device handle.
 *
 * @return
 */
PN_API PnStatus pnCloseDevice(int device);

/**
 * Turn output voltages (vSupply and vTune) on and off.
 *
 * @param[in] device Device handle.
 * 
 * @param[in] enabled Enables or disables output voltages.
 *
 * @return
 */
PN_API PnStatus pnEnableOutput(int device, bool enabled);

/**
 * Specify the output supply voltage.
 *
 * @param[in] device Device handle.
 * 
 * @param[in] voltage The value of vSupply in volts.
 *
 * @return
 */
PN_API PnStatus pnSetVcc(int device, double voltage);

/**
 * Specify the output tune voltage.
 *
 * @param[in] device Device handle.
 * 
 * @param[in] voltage The value of vTune in volts.
 *
 * @return
 */
PN_API PnStatus pnSetVtune(int device, double voltage);

/**
 * Send a trigger.
 *
 * @param[in] device Device handle.
 * 
 * @return
 */
PN_API PnStatus pnTrigger(int device);

/**
 * Return operational information of a device.
 *
 * @param[in] device Device handle.
 *
 * @param[out] usbVoltage Voltage from USB in volts.
 *
 * @param[out] dcInputVoltage Voltage from DC input in volts.
 * 
 * @param[out] vcoVoltage Voltage read back at vSupply in volts.
 * 
 * @param[out] vcoCurrent Current read back at vSupply in amps.
 *
 * @return
 */
PN_API PnStatus pnGetDiagnostics(int device, double *usbVoltage, double *dcInputVoltage, double *vcoVoltage, double *vcoCurrent);

/**
 * Get the serial number and firmware version of an open device.
 *
 * @param[in] device Device handle.
 *
 * @param[out] serial Returns device serial number.
 * 
 * @param[out] firmware Returns device firmware version.
 *
 * @return
 */
PN_API PnStatus pnGetSerialAndFirmware(int device, uint32_t *serial, uint32_t *firmware);

/**
 * Get the API version.
 *
 * @return
 * The returned string is of the form
 *
 * major.minor.revision
 *
 * Ascii periods ('.') separate positive integers. Major/minor/revision are not
 * guaranteed to be a single decimal digit. The string is null terminated. The
 * string should not be modified or freed by the user. An example string is
 * below?
 *
 * ['3' | '.' | '0' | '.' | '1' | '1' | '\0'] = "3.0.11"
 */
PN_API const char *pnGetAPIVersion();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // #ifndef PN_API_H
