// Copyright (c) 2022 Signal Hound
// For licensing information, please see the API license in the software_licenses folder

/*!
 * \file sa_api.h
 * \brief API functions for the SA44/124 spectrum analyzers.
 *
 * This is the main file for user accessible functions for controlling the
 * SA44 and SA124 spectrum analyzers.
 *
 */

#ifndef __SA_API_H__
#define __SA_API_H__

#if defined(_WIN32)
    #ifdef SA_EXPORTS
        #define SA_API __declspec(dllexport)
    #else
        #define SA_API __declspec(dllimport)
    #endif
#else // Linux
    #define SA_API
#endif

#if defined(_WIN32)
    #define SA_DEPRECATED(msg) __declspec(deprecated(msg))
#elif  defined(__GNUC__)
    #define SA_DEPRECATED(msg) __attribute__((deprecated))
#else
    #define SA_DEPRECATED(msg) msg
#endif

/** Used for boolean true when integer parameters are being used. */
#define SA_TRUE (1)
/** Used for boolean false when integer parameters are being used. */
#define SA_FALSE (0)

/**
 * Maximum number of devices that can be interfaced in the API. See
 * #saGetSerialNumberList.
 */
#define SA_MAX_DEVICES 8

/**
 * Device type
 */
typedef enum saDeviceType {
    /** None */
    saDeviceTypeNone = 0,
    /** SA44 */
    saDeviceTypeSA44 = 1,
    /** SA44B */
    saDeviceTypeSA44B = 2,
    /** SA124A */
    saDeviceTypeSA124A = 3,
    /** SA124B */
    saDeviceTypeSA124B = 4
} saDeviceType;

/**
 * Minimum frequency (Hz) for sweeps, and minimum center frequency for I/Q
 * measurements for SA44 devices. See #saConfigCenterSpan.
 */
#define SA44_MIN_FREQ (1.0)
/**
 * Minimum frequency (Hz) for sweeps, and minimum center frequency for I/Q
 * measurements for SA124 devices. See #saConfigCenterSpan.
 */
#define SA124_MIN_FREQ (100.0e3)
/**
 * Maximum frequency (Hz) for sweeps, and maximum center frequency for I/Q
 * measurements for SA44 devices. See #saConfigCenterSpan.
 */
#define SA44_MAX_FREQ (4.4e9)
/**
 * Maximum frequency (Hz) for sweeps, and maximum center frequency for I/Q
 * measurements for SA124 devices. See #saConfigCenterSpan.
 */
#define SA124_MAX_FREQ (13.0e9)
/** Minimum span (Hz) for sweeps. See #saConfigCenterSpan. */
#define SA_MIN_SPAN (1.0)
/** Maximum reference level in dBm. See #saConfigLevel. */
#define SA_MAX_REF (20)
/** Maximum attentuation. Valid values [0,3] or -1 for auto. See #saConfigGainAtten. */
#define SA_MAX_ATTEN (3)
/** Maximum gain. Valid values [0,2] or -1 for auto. See #saConfigGainAtten. */
#define SA_MAX_GAIN (2)
/** Minimum RBW (Hz) for sweeps. See #saConfigSweepCoupling. */
#define SA_MIN_RBW (0.1)
/** Maximum RBW (Hz) for sweeps. See #saConfigSweepCoupling. */
#define SA_MAX_RBW (6.0e6)
/** Minimum RBW (Hz) for device configured in real-time measurement mode. See #saConfigSweepCoupling. */
#define SA_MIN_RT_RBW (100.0)
/** Maximum RBW (Hz) for device configured in real-time measurement mode. See #saConfigSweepCoupling. */
#define SA_MAX_RT_RBW (10000.0)
/** Minimum I/Q bandwidth (Hz). See #saConfigIQ. */
#define SA_MIN_IQ_BANDWIDTH (100.0)
/** Maximum I/Q bandwidth (Hz). See #saConfigIQ. */
#define SA_MAX_IQ_DECIMATION (128)

/** Base I/Q sample rate in Hz. See #saConfigIQ. */
#define SA_IQ_SAMPLE_RATE (486111.111)

/** Measurement mode: Idle, no measurement. See #saInitiate. */
#define SA_IDLE      (-1)
/** Measurement mode: Swept spectrum analysis. See #saInitiate. */
#define SA_SWEEPING  (0x0)
/** Measurement mode: Real-time spectrum analysis. See #saInitiate. */
#define SA_REAL_TIME (0x1)
/** Measurement mode: I/Q streaming. See #saInitiate. */
#define SA_IQ        (0x2)
/** Measurement mode: Audio demod. See #saInitiate. */
#define SA_AUDIO     (0x3)
/** Measurement mode: Tracking generator sweeps for scalar network analysis. See #saInitiate. */
#define SA_TG_SWEEP  (0x4)

/** Specifies the Stanford flattop window used for sweep and real-time analysis. See #saConfigRBWShape. */
#define SA_RBW_SHAPE_FLATTOP (0x1)
/** Specifies a Gaussian window with 6dB cutoff used for sweep and real-time analysis. See #saConfigRBWShape. */
#define SA_RBW_SHAPE_CISPR (0x2)

/** Use min/max detector for sweep and real-time spectrum analysis. See #saConfigAcquisition. */
#define SA_MIN_MAX (0x0)
/** Use average detector for sweep and real-time spectrum analysis. See #saConfigAcquisition. */
#define SA_AVERAGE (0x1)

/** Specifies dBm units of sweep and real-time spectrum analysis measurements. See #saConfigAcquisition. */
#define SA_LOG_SCALE      (0x0)
/** Specifies mV units of sweep and real-time spectrum analysis measurements. See #saConfigAcquisition. */
#define SA_LIN_SCALE      (0x1)
/** Specifies dBm units, with no corrections, of sweep and real-time spectrum analysis measurements. See #saConfigAcquisition. */
#define SA_LOG_FULL_SCALE (0x2) // N/A
/** Specifies mV units, with no corrections, of sweep and real-time spectrum analysis measurements. See #saConfigAcquisition. */
#define SA_LIN_FULL_SCALE (0x3) // N/A

/** Automatically choose attenuation based on reference level. See #saConfigGainAtten. */
#define SA_AUTO_ATTEN (-1)
/** Automatically choose gain based on reference level. See #saConfigGainAtten. */
#define SA_AUTO_GAIN  (-1)

/** VBW processing occurs in dBm. See #saConfigProcUnits. */
#define SA_LOG_UNITS   (0x0)
/** VBW processing occurs in linear voltage units (mV). See #saConfigProcUnits. */
#define SA_VOLT_UNITS  (0x1)
/** VBW processing occurs in linear power units (mW). See #saConfigProcUnits. */
#define SA_POWER_UNITS (0x2)
/** No VBW processing. See #saConfigProcUnits. */
#define SA_BYPASS      (0x3)

/** Audio demodulation type: AM. See #saConfigAudio. */
#define SA_AUDIO_AM  (0x0)
/** Audio demodulation type: FM. See #saConfigAudio. */
#define SA_AUDIO_FM  (0x1)
/** Audio demodulation type: Upper side band. See #saConfigAudio. */
#define SA_AUDIO_USB (0x2)
/** Audio demodulation type: Lower side band. See #saConfigAudio. */
#define SA_AUDIO_LSB (0x3)
/** Audio demodulation type: CW. See #saConfigAudio. */
#define SA_AUDIO_CW  (0x4)

/** In scalar network analysis, use the next trace as a thru. See #saStoreTgThru. */
#define TG_THRU_0DB  (0x1)
/** In scalar network analysis, improve accuracy with a second thru step. See #saStoreTgThru. */
#define TG_THRU_20DB  (0x2)

/** Additional corrections are applied to tracking generator timebase. See #saSetTgReference. */
#define SA_REF_UNUSED (0)
/** Use tracking generator timebase as frequency standard for system, and do not apply additional corrections. See #saSetTgReference. */
#define SA_REF_INTERNAL_OUT (1)
/** Use an external reference for TG124A, and do not apply additional corrections to timebase. See #saSetTgReference. */
#define SA_REF_EXTERNAL_IN (2)

/**
* Results of running a self test. See #saSelfTest.
*/
typedef struct saSelfTestResults {
    /** Band mixers pass/fail results. */
    bool highBandMixer, lowBandMixer;
    /** Attenuator, second IF, and preamplifier pass/fail results. */
    bool attenuator, secondIF, preamplifier;
    /** Band mixer readings. */
    double highBandMixerValue, lowBandMixerValue;
    /** Attenuator, second IF, and preamplifier readings. */
    double attenuatorValue, secondIFValue, preamplifierValue;
} saSelfTestResults;

/**
 * Used to encapsulate I/Q data and metadata. See #saGetIQData.
 */
typedef struct saIQPacket {
    /**
     * Pointer to an array of 32-bit complex floating-point values. Complex
     * values are interleaved real-imaginary pairs. This must point to a
     * contiguous block of _iqCount_ complex pairs.
     */
    float *iqData;
    /** Number of I/Q data pairs to return. */
    int iqCount;
    /**
     * Specifies whether to discard any samples acquired by the API since the
     * last time and #saGetIQData function was called. Set to #SA_TRUE if you
     * wish to discard all previously acquired data, and #SA_FALSE if you wish
     * to retrieve the contiguous I/Q values from a previous call to this
     * function.
     */
    int purge;
    /**  How many I/Q samples are still left buffered in the API. Set by API. */
    int dataRemaining;
    /**
     * Returns #SA_TRUE or #SA_FALSE. Will return #SA_TRUE when the API is
     * required to drop data due to internal buffers wrapping. This can be
     * caused by I/Q samples not being polled fast enough, or in instances
     * where the processing is not able to keep up (underpowered systems, or
     * other programs utilizing the CPU) Will return #SA_TRUE on the capture in
     * which the sample break occurs. Does not indicate which sample the break
     * occurs on. Will always return #SA_FALSE if purge is true. Set by API.
     */
    int sampleLoss;
    /**
     * Seconds since epoch representing the timestamp of the first sample in
     * the returned array. Set by API.
     */
    int sec;
    /**
     * Milliseconds representing the timestamp of the first sample in the
     * returned array. Set by API.
     */
    int milli;
} saIQPacket;

/**
 * Status code returned from all SA API functions.
 * Errors are negative and suffixed with 'Err'.
 * Errors stop the flow of execution, warnings do not.
 */
typedef enum saStatus {
    /** Unknown/unexpected error. */
    saUnknownErr = -666,

    // Setting specific error codes

    /** Span outside frequency range. */
    saFrequencyRangeErr = -99,
    /** Invalid detector value provided. */
    saInvalidDetectorErr = -95,
    /** Invalid scale provided. */
    saInvalidScaleErr = -94,
    /** Invalid resolution bandwidth provided: must be between 0.1 Hz and 4.4 GHz. */
    saBandwidthErr = -91,
    /** External Reference Not Found. Internal reference will be used. */
    saExternalReferenceNotFound = -89,

    // Device-specific errors

    /** LNA error. */
    saLNAErr = -21,
    /** 10 MHz OCXO cold. Try again after 60 second warm-up. */
    saOvenColdErr = -20,

    // Data errors

    /**
     * Unable to connect to the internet or unable to find the necessary
     * calibration file on the internet during initialization. Ensure your PC
     * has an internet connection and try again. If it is connected to the
     * internet and you are still receiving this message, please contact Signal
     * Hound.
     */
    saInternetErr = -12,
    /** USB communications error. */
    saUSBCommErr = -11,

    // General configuration errors

    /** Unable to find a connected and available Signal Hound tracking generator. */
    saTrackingGeneratorNotFound = -10,
    /** Cannot perform requested operation while the device is active. */
    saDeviceNotIdleErr = -9,
    /** Device not found. */
    saDeviceNotFoundErr = -8,
    /** Cannot perform the requested operation in this mode of operation. */
    saInvalidModeErr = -7,
    /** The device is not properly configured. */
    saNotConfiguredErr = -6,
    /** Unable to open any more devices. */
    saTooManyDevicesErr = -5,
    /** Invalid parameter provided. */
    saInvalidParameterErr = -4,
    /** Device specified is not open. */
    saDeviceNotOpenErr = -3,
    /** Invalid device number provided. */
    saInvalidDeviceErr = -2,
    /** One or more parameters are NULL. */
    saNullPtrErr = -1,

    /** Function returned successfully. No warnings or errors. */
    saNoError = 0,

    // Warnings

    /** No corrections found on device, data will be uncalibrated. */
    saNoCorrections = 1,
    /** Device in compression. IF Overload. */
    saCompressionWarning = 2,
    /**
     * Supplied parameter clamped/limited to a set of known values or to within
     * an accepted range of operating values.
     */
    saParameterClamped = 3,
    /** Supplied bandwidth limited to within an accepted range of operating values. */
    saBandwidthClamped = 4,
    /** Unable to store correction data locally. */
    saCalFilePermissions = 5,
} saStatus;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * This function returns the devices that are unopened in the current process.
 * Up to #SA_MAX_DEVICES devices will be returned. The serial numbers of the
 * unopened devices are returned. The provided array will be populated starting
 * at index 0. The integer pointed to by _deviceCount_ will equal the number of
 * devices reported by this function upon returning.
 *
 * @param[out] serialNumbers A pointer to an array of at minimum
 * #SA_MAX_DEVICES contiguous 32-bit integers. It is undefined behavior if the
 * array pointed to by _serialNumbers_ is not #SA_MAX_DEVICES integers in
 * length.
 *
 * @param[out] deviceCount Pointer to an integer. Will be set to the number of
 * devices found on the system.
 *
 * @return
 */
SA_API saStatus saGetSerialNumberList(int serialNumbers[8], int *deviceCount);

/**
 * This function is similar to #saOpenDevice except you can specify the serial
 * number of the device you wish to open. Everything else is identical.
 *
 * @param[out] device Pointer to 32-bit integer variable. If this function
 * returns successfully, the value _device_ points to will contain a unique
 * handle value to the device opened. This number is used for all successive
 * API function calls.
 *
 * @param[in] serialNumber User-provided serial number.
 *
 * @return
 */
SA_API saStatus saOpenDeviceBySerialNumber(int *device, int serialNumber);

/**
 * This function attempts to open the first SA44/SA124 it detects. If a device
 * is opened successfully, a handle to the device will be returned through the
 * _device_ pointer which can be used to target that device for other API
 * calls.
 *
 * When successful, this function takes about 5 seconds to perform. If it is
 * the first time a device is opened on a host PC, this function can
 * potentially take much longer. See @ref firstTimeOpening.
 *
 * @param[out] device Pointer to 32-bit integer variable. If this function
 * returns successfully, the value _device_ points to will contain a unique
 * handle value to the device opened. This number is used for all successive
 * API function calls.
 *
 * @return
 */
SA_API saStatus saOpenDevice(int *device);

/**
 * This function is called when you wish to terminate a connection with a
 * device. Any resources the device has allocated will be freed and the USB 2.0
 * connection to the device is terminated. The device closed will be released
 * and will become available to be opened again. Any activity the device is
 * performing is aborted automatically before closing.
 *
 * @param[in] device Device handle.
 *
 * @return
 */
SA_API saStatus saCloseDevice(int device);

/**
 * This function exists to invoke a hard reset of the device. This will
 * function similarly to a power cycle (unplug/re-connect the device). This
 * might be useful if the device has entered an undesirable or unrecoverable
 * state. This function might allow the software to perform the reset rather
 * than ask the user to perform a power cycle.
 *
 * Functionally, in addition to a hard reset, this function closes the device
 * as if #saCloseDevice was called. This means the device handle becomes
 * invalid and the device must be reopened for use.
 *
 * This function is a blocking call and takes about 2.5 seconds to return.
 *
 * ## Example
 *
 * ```c++
 * // This short function shows how to preset a device in the
 * // quickest way possible. Both saPreset and saOpenDevice
 * // are blocking calls. It may be preferred to perform this
 * // function in a separate thread.
 *
 * saStatus PresetDevice(int *device_id)
 * {
 *     saPreset(*device_id);
 *     return saOpenDevice(device_id);
 * }
 * ```
 *
 * @param[in] device Device handle.
 *
 * @return
 */
SA_API saStatus saPreset(int device);

/**
 * This function should be called before opening the device. Ideally it should
 * be the first function called in a program interfacing the device. The path
 * specified should be suffixed with the ‘/’ or ‘\\’ character, denoting that
 * it is a directory. Failure to append a slash character will result in
 * undesired behavior.
 *
 * See examples of usage below.
 *
 * When a device is opened, the API will look in the supplied path for the
 * required calibration files. If the folder does not exist, it will be
 * created. If the files do not exist in the folder, they will be acquired from
 * the device flash memory where applicable, and for all other devices
 * downloaded from the Signal Hound web server. The files will be placed in the
 * supplied directory for future use. If the files exist in the directory, they
 * will simply be loaded into memory.
 *
 * __This function does not need to be called, as it will use a default system
 * path, typically `C:/ProgramData/SignalHound/cal_files/`.__
 *
 * ## Examples
 *
 * To set the working directory as the calibration file path call the function
 * as `saSetCalFilePath(“.\\”);`
 *
 * To set an absolute path, you might call the function as
 * `saSetCalFilePath(“C:\\ProgramData\\SignalHound\\CalFiles\\”);`
 *
 * @param[in] path
 *
 * @return
 */
SA_API saStatus saSetCalFilePath(const char *path);

/**
 * This function may be called only after the device has been opened. The
 * serial number returned should match the number on the case.
 *
 * @param[in] device Device handle.
 *
 * @param[out] serial Pointer to a 32-bit integer which will be assigned the
 * serial number of the device specified.
 *
 * @return
 */
SA_API saStatus saGetSerialNumber(int device, int *serial);

/**
 * Use this function to determine the firmware version of a specified device.
 *
 * @param[in] device Device handle.
 *
 * @param[out] firmwareString Pointer to a char array. The array should be at
 * minimum 16 chars in length.
 *
 * @return
 */
SA_API saStatus saGetFirmwareString(int device, char firmwareString[16]);

/**
 * This function may be called only after the device has been opened. If the
 * device handle is valid, _type_ will contain the model type of the device
 * pointed to by the handle. _type_ is an enumerated value of type
 * #saDeviceType.
 *
 * @param[in] device Device handle.
 *
 * @param[out] device_type Pointer to an integer to receive the model type.
 *
 * @return
 */
SA_API saStatus saGetDeviceType(int device, saDeviceType *device_type);

/**
 * _detector_ specifies how to produce the results of the signal processing for
 * the final sweep. Depending on settings, potentially many overlapping FFTs
 * will be performed on the input time domain data to retrieve a more
 * consistent and accurate final result. When the results overlap detector
 * chooses whether to average the results together, or maintain the minimum and
 * maximum values. If averaging is chosen, the _min_ and _max_ sweep arrays
 * will contain the same averaged data.
 *
 * The _scale_ parameter will change the units of returned sweeps. If
 * #SA_LOG_SCALE is provided sweeps will be returned in amplitude unit dBm. If
 * #SA_LIN_SCALE, the returned units will be in millivolts. If the full scale
 * units are specified, no corrections are applied to the data and amplitudes
 * are taken directly from the full scale input
 *
 * @param[in] device Device handle.
 *
 * @param[in] detector Specifies the video detector. The two possible values
 * are #SA_MIN_MAX and #SA_AVERAGE
 *
 * @param[in] scale Specifies the scale in which sweep results are returned.
 * The four possible values are #SA_LOG_SCALE, #SA_LIN_SCALE,
 * #SA_LOG_FULL_SCALE, and #SA_LIN_FULL_SCALE.
 *
 * @return
 */
SA_API saStatus saConfigAcquisition(int device, int detector, int scale);

/**
 * This function configures the operating frequency band of the device. Start
 * and stop frequencies can be determined from the _center_ and _span_.
 *
 * `start = center – (span / 2)`
 * `stop = center + (span / 2)`
 *
 * The values provided are used by the device during initialization and a more
 * precise start frequency is returned after initiation. Refer to
 * #saQuerySweepInfo for more information.
 *
 * Each device has a specified operational frequency range between some minimum
 * and maximum frequency. The limits are defined by #SA44_MIN_FREQ,
 * #SA124_MIN_FREQ, #SA44_MAX_FREQ, and #SA124_MAX_FREQ. The _center_ and
 * _span_ provided cannot specify a sweep outside of this range. Certain modes
 * of operation have specific frequency range limits. Those mode-dependent
 * limits are tested against during #saInitiate and not here.
 *
 * @param[in] device Device handle.
 *
 * @param[in] center Center frequency in hertz.
 *
 * @param[in] span Span in hertz.
 *
 * @return
 */
SA_API saStatus saConfigCenterSpan(int device, double center, double span);

/**
 * This function is best utilized when the device attenuation and gain are set
 * to automatic (default). When both attenuation and gain are set to
 * #SA_AUTO_ATTEN and #SA_AUTO_GAIN, respectively, the API uses the reference
 * level to best choose the gain and attenuation for maximum dynamic range. The
 * API chooses attenuation and gain values best for analyzing signal at or
 * below the reference level. For this reason, to achieve the best results, use
 * auto gain and atten, and set your reference level at or slightly about your
 * expected input power for best sensitivity. Reference level is specified in
 * dBm units.
 *
 * @param[in] device Device handle.
 *
 * @param[in] ref Reference level in dBm.
 *
 * @return
 */
SA_API saStatus saConfigLevel(int device, double ref);

/**
 * To set attenuation or gain to automatic, pass #SA_AUTO_GAIN and
 * #SA_AUTO_ATTEN as parameters. The preamp parameter is ignored when gain and
 * attenuation are automatic and is chosen automatically.
 *
 * ## _atten_ parameter
 *
 * | Supplied Parameter | SA44 Attenuation | SA124 Attenuation |
 * |--------------------|------------------|-------------------|
 * | SA_AUTO_ATTEN (-1) | Auto             | Auto              |
 * | 0                  | 0 dB             | 0 dB              |
 * | 1                  | 5 dB             | 10 dB             |
 * | 2                  | 10 dB            | 20 dB             |
 * | 3                  | 15 dB            | 30 dB             |
 *
 * ## _gain_ parameter
 *
 * | Supplied Parameter | Gain               |
 * |--------------------|--------------------|
 * | SA_AUTO_GAIN (-1)  | Auto               |
 * | 0                  | 16 dB Attenuation  |
 * | 1                  | Mid-Range          |
 * | 2                  | 12 dB Digital Gain |
 *
 * By default, if this function is not called, gain and attenuation are set to
 * automatic. It is suggested to leave these values as automatic as it will
 * greatly increase the consistency of your results. If you choose to manually
 * control gain and attenuation please read @ref gainAndAtten.
 *
 * @param[in] device Device handle.
 *
 * @param[in] atten Attenuator setting.
 *
 * @param[in] gain Gain setting.
 *
 * @param[in] preAmp Specify whether to enable the internal device
 * pre-amplifier.
 *
 * @return
 */
SA_API saStatus saConfigGainAtten(int device, int atten, int gain, bool preAmp);

/**
 * The resolution bandwidth, or RBW, represents the bandwidth of spectral
 * energy represented in each frequency bin. For example, with an RBW of 10
 * kHz, the amplitude value for each bin would represent the total energy from
 * 5 kHz below to 5 kHz above the bin’s center.
 *
 * The video bandwidth, or VBW, is applied after the signal has been converted
 * to frequency domain as power, voltage, or log units. It is implemented as a
 * simple rectangular window, averaging the amplitude readings for each
 * frequency bin over several overlapping FFTs. A signal whose amplitude is
 * modulated at a much higher frequency than the VBW will be shown as an
 * average, whereas amplitude modulation at a lower frequency will be shown as
 * a minimum and maximum value.
 *
 * Available RBWs are [0.1Hz – 100kHz] and 250kHz. For the SA124 devices, a
 * 6MHz RBW is available as well. Not all RBWs will be available depending on
 * span, for example the API may restrict RBW when a sweep size exceeds a
 * certain amount. Also there are many hardware limitations that restrict
 * certain RBWs, for a full list of these restrictions, see @ref rbwAndVbw.
 *
 * The parameter _reject_ determines whether software image reject will be
 * performed. The SA series spectrum analyzers do not have hardware-based image
 * rejection, instead relying on a software algorithm to reject image
 * responses. See the
 * [SA44B User Manual](https://signalhound.com/sigdownloads/SA44B/SA44B-User-Manual.pdf)
 * or
 * [SA124B User Manual](https://signalhound.com/sigdownloads/SA124B/SA124B-User-Manual.pdf)
 * for additional details. Generally, set _reject_ to true for continuous
 * signals, and false to catch short duration signals at a known frequency. To
 * capture short duration signals with an unknown frequency, consider the
 * Signal Hound [BB60C](https://signalhound.com/products/bb60c/),
 * [BB60D](https://signalhound.com/products/bb60d-6-ghz-real-time-spectrum-analyzer/),
 * [SM200B](https://signalhound.com/products/sm200b-20-ghz-real-time-spectrum-analyzer/),
 * [SM200C](https://signalhound.com/products/sm200c-20-ghz-real-time-spectrum-analyzer-with-10gbe/),
 * or
 * [SM435B](https://signalhound.com/products/sm435b-43-5-ghz-real-time-spectrum-analyzer/)
 * spectrum analyzers.
 *
 * @param[in] device Device handle.
 *
 * @param[in] rbw Resolution bandwidth in Hz. RBW can be arbitrary.
 *
 * @param[in] vbw Video bandwidth in Hz. VBW must be less than or equal to RBW.
 * VBW can be arbitrary. For best performance use RBW as the VBW.
 *
 * @param[in] reject Indicates whether to enable image rejection.
 *
 * @return
 */
SA_API saStatus saConfigSweepCoupling(int device, double rbw, double vbw, bool reject);

/**
 * Specify the RBW filter shape, which is achieved by changing the window
 * function. When specifying #SA_RBW_SHAPE_FLATTOP, a custom bandwidth flat-top
 * window is used measured at the 3dB cutoff point. When specifying
 * #SA_RBW_SHAPE_CISPR, a Gaussian window with zero-padding is used to achieve
 * the specified RBW. The Gaussian window is measured at the 6dB cutoff point.
 *
 * @param[in] device Device handle.
 *
 * @param[in] rbwShape An acceptable RBW filter shape value, either
 * #SA_RBW_SHAPE_FLATTOP or #SA_RBW_SHAPE_CISPR.
 *
 * @return
 */
SA_API saStatus saConfigRBWShape(int device, int rbwShape);

/**
 * The units provided determines what unit type video processing occurs in. The
 * chart below shows which unit types are used for each units selection.
 *
 * For “average power” measurements, #SA_POWER_UNITS should be selected. For
 * cleaning up an amplitude modulated signal, #SA_VOLT_UNITS would be a good
 * choice. To emulate a traditional spectrum analyzer, select #SA_LOG_UNITS. To
 * minimize processing power and bypass video bandwidth processing, select
 * #SA_BYPASS.
 *
 * | Macro          | Unit                |
 * |----------------|---------------------|
 * | SA_LOG_UNITS   | dBm                 |
 * | SA_VOLT_UNITS  | mV                  |
 * | SA_POWER_UNITS | mW                  |
 * | SA_BYPASS      | No video processing |
 *
 * @param[in] device Device handle.
 *
 * @param[in] units The possible values are #SA_POWER_UNITS, #SA_LOG_UNITS,
 * #SA_VOLT_UNITS, and #SA_BYPASS.
 *
 * @return
 */
SA_API saStatus saConfigProcUnits(int device, int units);

/**
 * This function is used to configure the digital I/Q data stream. A decimation
 * factor and filter bandwidth are able to be specified. The decimation rate
 * divides the I/Q sample rate directly while the bandwidth parameter further
 * filters the digital stream.
 *
 * For any given decimation rate, a minimum filter bandwidth must be applied to
 * account for sufficient filter roll off. If a bandwidth value is supplied
 * above the maximum for a given decimation, the bandwidth will be clamped to
 * the maximum value. For a list of possible decimation values and associated
 * bandwidth values, see the table below.
 *
 * The base sample rate of the SA44 and SA124 spectrum analyzers is 486.111111
 * (repeating) kS/s. To get a precise sample rate given a decimation value, use
 * this equation.
 *
 * `sample rate = 486111.11111~ / decimation`
 *
 * | Decimation Rate | Maximum Bandwidth |
 * |-----------------|-------------------|
 * | 1               | 250.0 kHz         |
 * | 2               | 225.0 kHz         |
 * | 4               | 100.0 kHz         |
 * | 8               | 50.0 kHz          |
 * | 16              | 20 kHz            |
 * | 32              | 12.0 kHz          |
 * | 64              | 5.0 kHz           |
 * | 128             | 3.0 kHz           |
 *
 * @param[in] device Device handle.
 *
 * @param[in] decimation Specify a decimation rate for the I/Q data stream.
 *
 * @param[in] bandwidth Specify the band pass filter width on the I/Q digital
 * stream.
 *
 * @return
 */
SA_API saStatus saConfigIQ(int device, int decimation, double bandwidth);

/**
 * This function is used to configure the majority of the audio stream
 * settings. A number of audio modulation types are supported, and a number of
 * filter parameters can be set.
 *
 * Below is the overall flow of data through our audio processing algorithm:
 *
 * ![](img/audio_demod_data_flow.png)
 *
 * @param[in] device Device handle.
 *
 * @param audioType Specifies the demodulation scheme, possible values are
 * #SA_AUDIO_AM, #SA_AUDIO_FM, #SA_AUDIO_USB, #SA_AUDIO_LSB, and #SA_AUDIO_CW.
 *
 * @param centerFreq Center frequency in Hz of audio signal to demodulate.
 *
 * @param bandwidth Intermediate frequency bandwidth centered on freq. Filter
 * takes place before demodulation. Specified in Hz. Should be between 500Hz
 * and 500kHz.
 *
 * @param audioLowPassFreq Post demodulation filter in Hz. Should be between
 * 1kHz and 12kHz.
 *
 * @param audioHighPassFreq Post demodulation filter in Hz. Should be between
 * 20 and 1000Hz.
 *
 * @param fmDeemphasis Specified in microseconds. Should be between 1 and 100.
 * This value is ignored if _audioType_ is not equal to #SA_AUDIO_FM.
 *
 * @return
 */
SA_API saStatus saConfigAudio(int device, int audioType, double centerFreq,
                              double bandwidth, double audioLowPassFreq,
                              double audioHighPassFreq, double fmDeemphasis);

/**
 * The function allows you to configure additional parameters of the real-time
 * frames returned from the API. If this function is not called a scale of
 * 100dB is used and a frame rate of 30fps is used. For more information
 * regarding real-time mode see @ref realTimeAnalysis.
 *
 * @param[in] device Device handle.
 *
 * @param[in] frameScale Specify the real-time frame height in dB. Values can
 * be between [10 - 200].
 *
 * @param[in] frameRate Specify the real-time frame rate in frames per seconds.
 * Values can be between [4 – 30].
 *
 * @return
 */
SA_API saStatus saConfigRealTime(int device, double frameScale, int frameRate);

/**
 * By setting the advance rate users can control the overlap rate of the FFT
 * processing in real-time spectrum analysis. The _advanceRate_ parameter
 * specifies how far the FFT window slides through the data for each FFT as a
 * function of FFT size. An _advanceRate_ of 0.5 specifies that the FFT window
 * will advance 50% the FFT length for each FFT for a 50% overlap rate.
 * Specifying a value of 1.0 would mean the FFT window advances the full FFT
 * length meaning there is no overlap in real-time processing. The default
 * value is 0.125 and the range of acceptable values is between [0.125, 10].
 * Increasing the advance rate reduces processing considerably but also
 * increases the 100% probability of intercept of the device.
 *
 * @param[in] device Device handle.
 *
 * @param[in] advanceRate FFT advance rate. See description.
 *
 * @return
 */
SA_API saStatus saConfigRealTimeOverlap(int device, double advanceRate);

/**
 * Configure the time base reference port for the device. By passing a value of
 * #SA_REF_INTERNAL_OUT you can output the internal 10MHz time base of the
 * device out on the reference port. By passing a value of #SA_REF_EXTERNAL_IN
 * the API attempts to enable a 10MHz reference on the reference BNC port. If
 * no reference is found, the device continues to use the internal reference
 * clock. Once a device has successfully switched to an external reference it
 * must remain using it until the device is closed, and it is undefined
 * behavior to disconnect the reference input from the reference BNC port.
 *
 * @param[in] device Device handle.
 *
 * @param[in] timebase Time base setting value. Acceptable inputs are
 * #SA_REF_INTERNAL_OUT and #SA_REF_EXTERNAL_IN.
 *
 * @return
 */
SA_API saStatus saSetTimebase(int device, int timebase);

/**
 * This function configures the device into a state determined by the _mode_
 * parameter. For more information regarding operating states, refer to the
 * @ref theoryOfOperation and @ref modesOfOperation sections. This function
 * calls #saAbort before attempting to reconfigure. It should be noted, if an
 * error occurs attempting to configure the device, any past operating state
 * will no longer be active and the device will become idle.
 *
 * @param[in] device Device handle.
 *
 * @param[in] mode The possible values for mode are #SA_IDLE, #SA_SWEEPING,
 * #SA_REAL_TIME, #SA_IQ, #SA_AUDIO, and #SA_TG_SWEEP.
 *
 * @param[in] flag This value is currently unused. Pass 0 as a parameter.
 *
 * @return
 */
SA_API saStatus saInitiate(int device, int mode, int flag);

/**
 * Stops the device operation and places the device into an idle state. If the
 * device is currently idle, then the function returns normally and returns
 * #saNoError.
 *
 * @param[in] device Device handle.
 *
 * @return
 */
SA_API saStatus saAbort(int device);

/**
 * This function should be called to determine sweep characteristics after a
 * device has been configured and initiated.
 *
 * @param[in] device Device handle.
 *
 * @param[out] sweepLength A pointer to a 32-bit integer. If the function
 * returns successfully, the integer pointed to will contain the size of the
 * _min_ and _max_ arrays returned by #saGetSweep_32f and #saGetSweep_64f.
 *
 * @param[out] startFreq A pointer to a 64-bit floating point variable. If the
 * function returns successfully, the variable _startFreq_ points to will equal
 * the frequency of the first bin in the configured sweep.
 *
 * @param[out] binSize A pointer to a 64-bit floating point variable. If the
 * function returns successfully, the variable _binSize_ points to will contain
 * the frequency difference between each bin in the configured sweep.
 *
 * @return
 */
SA_API saStatus saQuerySweepInfo(int device, int *sweepLength, double *startFreq, double *binSize);

/**
 * Use this function to get the parameters of the I/Q data stream.
 *
 * @param[in] device Device handle.
 *
 * @param[out] returnLen Pointer to a 32-bit integer. If the function returns
 * successfully, the variable _returnLen_ points to will contain the number of
 * I/Q sample pairs that will be returned by calling #saGetIQ_32f/#saGetIQ_64f.
 *
 * @param[out] bandwidth Pointer to a 64-bit float. If the function returns
 * successfully, the variable _bandwidth_ points to will contain the bandpass
 * filter bandwidth width in Hz. Width is specified by the 3dB roll-off points.
 *
 * @param[out] samplesPerSecond Pointer to a 32-bit integer. If the function
 * returns successfully, the variable _samplesPerSecond_ points to will contain
 * the sample rate of the configured I/Q data stream.
 *
 * @return
 */
SA_API saStatus saQueryStreamInfo(int device, int *returnLen, double *bandwidth, double *samplesPerSecond);

/**
 * This function should be called after initializing the device for real-time
 * mode. This device returns the frame size of the real-time frame configured.
 *
 * @param[in] device Device handle.
 *
 * @param[out] frameWidth Pointer to a 32-bit signed integer representing the
 * width of the real-time frame.
 *
 * @param[out] frameHeight Pointer to a 32-bit signed integer representing the
 * height of the real-time frame.
 *
 * @return
 */
SA_API saStatus saQueryRealTimeFrameInfo(int device, int *frameWidth, int *frameHeight);

/**
 * When this function returns successfully, the value _poi_ points to will
 * contain the 100% probability of intercept duration in seconds of the device
 * as currently configured in real-time spectrum analysis. The device must
 * actively be configured and initialized in the real-time spectrum analysis
 * mode.
 *
 * @param[in] device Device handle.
 *
 * @param[out] poi Pointer to double. See description.
 *
 * @return
 */
SA_API saStatus saQueryRealTimePoi(int device, double *poi);

/**
 * Upon returning successfully, this function returns the _minimum_ and
 * _maximum_ floating point arrays of one full sweep. If the _detector_
 * provided in #saConfigAcquisition is #SA_AVERAGE, the arrays will be
 * populated with the same values. Element zero of each array corresponds to
 * the _startFreq_ returned from #saQuerySweepInfo.
 *
 * @param[in] device Device handle.
 *
 * @param[out] min Pointer to the beginning of an array of 32-bit floating
 * point values, whose length is equal to or greater than _sweepLength_
 * returned from #saQuerySweepInfo.
 *
 * @param[out] max Pointer to the beginning of an array of 32-bit floating
 * point values, whose length is equal to or greater than _sweepLength_
 * returned from #saQuerySweepInfo.
 *
 * @return
 */
SA_API saStatus saGetSweep_32f(int device, float *min, float *max);

/**
 * See #saGetSweep_32f.
 *
 * @param[in] device Device handle.
 *
 * @param[out] min Pointer to the beginning of an array of 64-bit floating
 * point values, whose length is equal to or greater than _sweepLength_
 * returned from #saQuerySweepInfo.
 *
 * @param[out] max Pointer to the beginning of an array of 64-bit floating
 * point values, whose length is equal to or greater than _sweepLength_
 * returned from #saQuerySweepInfo.
 *
 * @return
 */
SA_API saStatus saGetSweep_64f(int device, double *min, double *max);

/**
 * This function is similar to the #saGetSweep_32f/saGetSweep_64f functions
 * except it can return in certain contexts before the full sweep is ready.
 * This function might return for sweeps which are going to take multiple
 * seconds, providing you with only a portion of the results. This might be
 * useful if you want to perform some signal analysis on portions of the sweep
 * as it is being received, or update other portions of your application during
 * a long acquisition process. Subsequent calls will always provide the next
 * contiguous portion of spectrum.
 *
 * A buffer that can hold the full sweep must be provided. The pointers _start_
 * and _stop_ are used to determine which portion of the sweep was updated. The
 * elements in the arrays from [_start_, _stop_] will be updated. _start_ and
 * `_stop_ - 1` can be used to index the updated portion of the arrays.
 *
 * The updated portion of the sweep will always at maximum end at the final
 * element of the sweep. For example, if only 20 frequency bins remain after
 * the previous call to #saGetPartialSweep_32f/#saGetPartialSweep_32f, the next
 * call will update at most 20 points. When the final portion of the sweep has
 * been updated, _stop_ will equal the sweep length. Calling this function
 * again will request and begin the next sweep.
 *
 * @param[in] device Device handle.
 *
 * @param min Pointer to an array of 32-bit floating point values, whose length
 * is equal to or greater than _sweepLength_ returned from #saQuerySweepInfo.
 *
 * @param max Pointer to an array of 32-bit floating point values, whose length
 * is equal to or greater than _sweepLength_ returned from #saQuerySweepInfo.
 *
 * @param start Pointer to a 32-bit integer variable. If the function returns
 * successfully, the variable _start_ points to will contain the start index of
 * the updated portion of the sweep.
 *
 * @param stop Pointer to a 32-bit integer variable. If the function returns
 * successfully, the variable _stop_ points to will contain the `index + 1` of
 * the last update value in the updated portion of the sweep.
 *
 * @return
 */
SA_API saStatus saGetPartialSweep_32f(int device, float *min, float *max, int *start, int *stop);

/**
 * See #saGetPartialSweep_32f.
 *
 * @param[in] device Device handle.
 *
 * @param min Pointer to an array of 64-bit floating point values, whose length
 * is equal to or greater than _sweepLength_ returned from #saQuerySweepInfo.
 *
 * @param max Pointer to an array of 64-bit floating point values, whose length
 * is equal to or greater than _sweepLength_ returned from #saQuerySweepInfo.
 *
 * @param start Pointer to a 32-bit integer variable. If the function returns
 * successfully, the variable _start_ points to will contain the start index of
 * the updated portion of the sweep.
 *
 * @param stop Pointer to a 32-bit integer variable. If the function returns
 * successfully, the variable _stop_ points to will contain the `index + 1` of
 * the last update value in the updated portion of the sweep.
 *
 * @return
 */
SA_API saStatus saGetPartialSweep_64f(int device, double *min, double *max, int *start, int *stop);

/**
 * This function is used to retrieve the real-time sweeps, frame, and alpha
 * frame for a measurement interval. This function should be used instead of
 * #saGetSweep_32f/#saGetSweep_64f and
 * #saGetPartialSweep_32f/#saGetPartialSweep_64f for real-time mode. The sweep
 * array should be ‘N’ contiguous floats, where N is the sweep length returned
 * from #saQuerySweepInfo. The _frame_ and _alphaFrame_ should be WxH values
 * long where W and H are the values returned from #saQueryRealTimeFrameInfo.
 * For more information see @ref realTimeAnalysis.
 *
 * @param[in] device Device handle.
 *
 * @param[out] minSweep If this pointer is non-null, the min held sweep will be
 * returned to the user. If the detector is set to average, this array will be
 * identical to the _maxSweep_ array returned.
 *
 * @param[out] maxSweep If this pointer is non-null, the max held sweep will be
 * returned to the user. If the detector is set to average, this array contains
 * the averaged results over the measurement interval.
 *
 * @param[out] colorFrame Pointer to a 32-bit floating point array. If the
 * function returns successfully, the contents of the array will contain a
 * single real-time frame.
 *
 * @param[out] alphaFrame Pointer to a 32-bit floating point array. If the
 * function returns successfully, the contents of the array will contain a
 * single real-time alpha frame.
 *
 * @return
 */
SA_API saStatus saGetRealTimeFrame(int device, float *minSweep, float *maxSweep, float *colorFrame, float *alphaFrame);

/**
 * Retrieve the next array of I/Q samples in the stream. The length of the
 * buffer provided to this function is the return length from
 * #saQueryStreamInfo * 2. #saQueryStreamInfo returns the length as I/Q sample
 * pairs. This function will need to be called ~30 times per second for any
 * given decimation rate for the internal circular buffers not to fall behind.
 * We recommend polling this function from a separate thread and not performing
 * any other tasks on the polling thread to ensure the thread does not fall
 * behind.
 *
 * The buffer will be populated with alternating I/Q sample pairs scaled to mW.
 * The time difference between each sample can be determined from the sample
 * rate of the configured device.
 *
 * @param[in] device Device handle.
 *
 * @param[out] iq A pointer to a 32-bit floating point array. The contents of
 * this buffer will be updated with interleaved I/Q digital samples.
 *
 * @return
 */
SA_API saStatus saGetIQ_32f(int device, float *iq);

/**
 * See #saGetIQ_32f.
 *
 * @param[in] device Device handle.
 *
 * @param[out] iq A pointer to a 64-bit floating point array. The contents of
 * this buffer will be updated with interleaved I/Q digital samples.
 *
 * @return
 */
SA_API saStatus saGetIQ_64f(int device, double *iq);

/**
 * This function retrieves one block of I/Q data as specified by the #saIQPacket
 * struct.
 *
 * @param[in] device Device handle.
 *
 * @param[out] pkt Pointer to a #saIQPacket struct.
 *
 * @return
 */
SA_API saStatus saGetIQData(int device, saIQPacket *pkt);

/**
 * This function provides an alternate interface to the #saGetIQData function.
 * Using this function, you can eliminate the need for the #saIQPacket struct,
 * which eases development of non-C programming language bindings (languages
 * and environments such as LabVIEW, MATLAB, Python, C#, etc).
 *
 * The parameters to this function have a one-to-one mapping to the members of
 * the #saIQPacket struct.
 *
 * This function is implemented by populating the #saIQPacket struct with the
 * provided parameters and calling #saGetIQData.
 *
 * @param[in] device Device handle.
 *
 * @param[out] iqData See #saIQPacket.
 *
 * @param[out] iqCount See #saIQPacket.
 *
 * @param[out] purge See #saIQPacket.
 *
 * @param[out] dataRemaining See #saIQPacket.
 *
 * @param[out] sampleLoss See #saIQPacket.
 *
 * @param[out] sec See #saIQPacket.
 *
 * @param[out] milli See #saIQPacket.
 *
 * @return
 */
SA_API saStatus saGetIQDataUnpacked(int device, float *iqData, int iqCount, int purge,
                                    int *dataRemaining, int *sampleLoss, int *sec, int *milli);

/**
 * If the device is initiated and running in the audio demodulation mode, the
 * function is a blocking call which returns the next 4096 audio samples. The
 * approximate blocking time for this function is 128 ms if called again
 * immediately after returning. There is no internal buffering of audio,
 * meaning the audio will be overwritten if this function is not called in a
 * timely fashion. The audio values are typically -1.0 to 1.0, representing
 * full-scale audio. In FM mode, the audio values will scale with a change in
 * IF bandwidth.
 *
 * @param[in] device Device handle.
 *
 * @param[out] audio Pointer to a 32-bit floating point value array.
 *
 * @return
 */
SA_API saStatus saGetAudio(int device, float *audio);

/**
 * Requesting the internal temperature of the device cannot be performed while
 * the device is currently active. To receive the absolute current internal
 * device temperature, ensure the device is inactive by calling #saAbort before
 * calling this function. If the device is active, the temperature returned
 * will be the last temperature returned from this function.
 *
 * @param[in] device Device handle.
 *
 * @param[out] temp Pointer to a 32-bit floating point value. If the function
 * returns successfully, the value _temp_ points to will contain the current
 * device temperature.
 *
 * @return
 */
SA_API saStatus saQueryTemperature(int device, float *temp);

/**
 * A USB voltage below 4.55V may cause readings to be out of spec. Check your
 * cable for damage and USB connectors for damage or oxidation.
 *
 * @param[in] device Device handle.
 *
 * @param[out] voltage Pointer to a 32-bit floating point value. If the
 * function returns successfully the variable _voltage_ points to will contain
 * the current voltage of the system.
 *
 * @return
 */
SA_API saStatus saQueryDiagnostics(int device, float *voltage);

/**
 * This function attempts to pair an unclaimed Signal Hound tracking generator
 * with an open Signal Hound spectrum analyzer.
 *
 * @param[in] device Device handle.
 *
 * @return
 */
SA_API saStatus saAttachTg(int device);

/**
 * This function is a helper function to determine if a tracking generator has
 * been previously paired with the specified device.
 *
 * @param[in] device Device handle.
 *
 * @param[out] attached Pointer to a boolean variable. If this function returns
 * successfully, the variable _attached_ points to will contain a true/false
 * value as to whether a tracking generator is paired with the spectrum
 * analyzer.
 *
 * @return
 */
SA_API saStatus saIsTgAttached(int device, bool *attached);

/**
 * This function configures the tracking generator sweeps. Through this
 * function you can request a sweep size. The sweep size is the number of
 * discrete points returned in the sweep over the configured span. The final
 * value chosen by the API can be different than the requested size by a factor
 * of 2 at most. The dynamic range of the sweep is determined by the choice of
 * _highDynamicRange_ and _passiveDevice_. A value of true for both provides
 * the highest dynamic range sweeps. Choosing false for _passiveDevice_
 * suggests to the API that the device under test is an active device
 * (amplification).
 *
 * @param[in] device Device handle.
 *
 * @param[in] sweepSize Suggested sweep size.
 *
 * @param[in] highDynamicRange Request the ability to perform two store thrus
 * for an increased dynamic range sweep.
 *
 * @param[in] passiveDevice Specify whether the device under test is a passive
 * device (no gain).
 *
 * @return
 */
SA_API saStatus saConfigTgSweep(int device, int sweepSize, bool highDynamicRange, bool passiveDevice);

/**
 * This function, with flag set to #TG_THRU_0DB, notifies the API to use the
 * last trace as a thru (your 0 dB reference). Connect your tracking generator
 * RF output to your spectrum analyzer RF input. This can be accomplished using
 * the included SMA to SMA adapter, or anything else you want the software to
 * establish as the 0 dB reference (e.g. the 0 dB setting on a step attenuator,
 * or a 20 dB attenuator you will be including in your amplifier test setup).
 *
 * After you have established your 0 dB reference, a second step may be
 * performed to improve the accuracy below -40 dB. With approximately 20-30 dB
 * of insertion loss between the spectrum analyzer and tracking generator, call
 * #saStoreTgThru with flag set to #TG_THRU_20DB. This corrects for slight
 * variations between the high gain and low gain sweeps.
 *
 * @param[in] device Device handle.
 *
 * @param[in] flag Specify the type of store thru. Possible values are
 * #TG_THRU_0DB and #TG_THRU_20DB.
 *
 * @return
 */
SA_API saStatus saStoreTgThru(int device, int flag);

/**
 * This function sets the output frequency and amplitude of the tracking
 * generator. This can only be performed if a tracking generator is paired with
 * a spectrum analyzer and is currently not configured and initiated for TG
 * sweeps.
 *
 * @param[in] device Device handle.
 *
 * @param[in] frequency Set the frequency, in Hz, of the TG output.
 *
 * @param[in] amplitude Set the amplitude, in dBm, of the TG output.
 *
 * @return
 */
SA_API saStatus saSetTg(int device, double frequency, double amplitude);

/**
 * Configure the time base for the tracking generator attached to the device
 * specified. When #SA_REF_UNUSED is specified additional frequency corrections
 * are applied. If using an external reference or you are using the TG time
 * base frequency as the frequency standard in your system, you will want to
 * specify #SA_REF_INTERNAL_OUT or #SA_REF_EXTERNAL_IN so the additional
 * corrections are not applied.
 *
 * @param[in] device Device handle.
 *
 * @param[in] reference A valid time base setting value. Possible values are
 * #SA_REF_UNUSED, #SA_REF_INTERNAL_OUT, and #SA_REF_EXTERNAL_IN.
 *
 * @return
 */
SA_API saStatus saSetTgReference(int device, int reference);

/**
 * Retrieve the last set TG output parameters the user set through the #saSetTg
 * function. The #saSetTg function must have been called for this function to
 * return valid values. If the TG was used to perform scalar network analysis
 * at any point, this function will not return valid values until the #saSetTg
 * function is called again.
 *
 * If a previously set parameter was clamped in the #saSetTg function, this
 * function will return the final clamped value.
 *
 * If any pointer parameter is null, that value is ignored and not returned.
 *
 * @param[in] device Device handle.
 *
 * @param[out] frequency The double variable that frequency points to will
 contain the last set frequency of the TG output in Hz.
 *
 * @param[out] amplitude The double variable that amplitude points to will
 contain the last set amplitude of the TG output in dBm.
 *
 * @return
 */
SA_API saStatus saGetTgFreqAmpl(int device, double *frequency, double *amplitude);

/**
 * The SA124A/B allows a user to configure the device to route the 6 MHz
 * bandwidth intermediate frequency directly to the IF output BNC port. While
 * the IF is routed to the BNC port, the device is incapable of performing
 * sweeps or I/Q streaming. There is no image rejection in this mode.
 *
 * Calling this function while the device is currently active in a different
 * mode will cause the API to abort the current mode of operation and enable
 * the IF output BNC port. To disable the IF output, simply call #saInitiate
 * with the new desired configuration.
 *
 * The local oscillator mixed with the RF must be 138 MHz or higher, so only
 * high side injection is available below 201 MHz.
 *
 * @param[in] device Device handle.
 *
 * @param[in] inputFreq The input center frequency on the SMA connector
 * specified in Hz. Must be between 125MHz and 13GHz.
 *
 * @param[in] outputFreq The desired output frequency on the BNC port specified
 * in Hz. Positive for low-side LO injection, negative for high-side. Must be
 * between 34 and 38MHz for the SA124A and between 61 and 65MHz for the SA124B.
 *
 * @param[in] inputAtten Attenuation of the input signal specified in dB. Must
 * be between 0 and 30 dB.
 *
 * @param[in] outputGain Amplification of the output signal specified in dB.
 * Must be between 0 and 60 dB.
 *
 * @return
 */
SA_API saStatus saConfigIFOutput(int device, double inputFreq, double outputFreq,
                                 int inputAtten, int outputGain);

/**
 * Performs a self-test and returns the results in an #saSelfTestResults
 * struct.
 *
 * @param[in] device Device handle.
 *
 * @param[out] results A pointer to an #saSelfTestResults struct.
 *
 * @return
 */
SA_API saStatus saSelfTest(int device, saSelfTestResults *results);

/**
 * Get the current API version.
 *
 * The returned string is of the form `major.minor.revision`.
 *
 * Ascii periods (“.”) separate positive integers. Major/Minor/Revision are
 * not gauranteed to be a single decimal digit. The string is null
 * terminated. An example string is below..
 *
 * `[ ‘1’ | ‘.’ | ‘2’ | ‘.’ | ‘1’ | ‘1’ | ‘\0’ ] = “1.2.11”`
 *
 * @return
 */
SA_API const char* saGetAPIVersion();

/**
 * The product ID is 4302-1103.
 *
 * @return
 */
SA_API const char* saGetProductID();

/**
 * Produce an ASCII string representation of a given status code. Useful for
 * debugging.
 *
 * @param[in] code A #saStatus value returned from an API call.
 *
 * @return
 */
SA_API const char* saGetErrorString(saStatus code);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SA_API_H
