using System;
using System.Runtime.InteropServices;
using System.Text;

/*
 * The C# API class for the SP series devices is a class of static members and methods which
 * are simply a 1-to-1 mapping of the C API. This makes is easy to modify and look up
 * functions in the API manual.
 */

enum SpStatus
{
    /** Internal use only. */
    spInternalFlashErr = -101,
    /** Internal use only. */
    spInternalFileIOErr = -100,

    /**
     * Unable to allocate memory for the measurement. Out of system memory.
     */
    spAllocationErr = -11,
    /** At maximum number of devices that can be interfaced. */
    spMaxDevicesConnectedErr = -10,
    /**
     * Often the result of trying to perform an action while the device is currently
     * making a measurement or not in an idle state, or performing an action
     * that is not supported by the current mode of operation. For instance,
     * requesting a sweep while the device is configured for I/Q streaming.
     */
    spInvalidConfigurationErr = -9,
    /**
     * For standard sweeps, this error indicates another sweep is already being
     * performed. You might encounter this in a multi-threaded application. For
     * queued sweep lists, this indicates the sweep at the given position is
     * already active. Finish this sweep before starting again.
     */
    spSweepAlreadyActiveErr = -8,
    /** Boot error */
    spBootErr = -7,
    /**
     * Indicates USB data framing issues. Data may be corrupt.
     * If error persists, reconfiguration/cycling might be required.
    */
    spFramingErr = -6,
    /**
     * Device disconnected. Will require the device to be closed and reopened
     * to continue. Most likely cause is large USB data loss or cable connectivity
     * issues.
     */
    spConnectionLostErr = -5,
    /** Invalid device handle specified. */
    spInvalidDeviceErr = -4,
    /** One or more required pointer parameters are null. */
    spNullPtrErr = -3,
    /** One or more required parameters found to have an invalid value. */
    spInvalidParameterErr = -2,
    /**
     * Unable to open device.
     * Verify the device is connected and the LED is solid green.
     */
    spDeviceNotFoundErr = -1,

    /** Function returned successfully. */
    spNoError = 0,

    /** One or more of the provided parameters were clamped to a valid range. */
    spSettingClamped = 1,
    /**
     * Temperature drift occured since last configuration.
     * Measurements might be uncalibrated. Reconfiguring the device when possible will
     * eliminate this warning.
     */
    spTempDrift = 2,
    /** Measurement includes data which caused an ADC overload (clipping/compression) */
    spADCOverflow = 3,
    /** Measurement is uncalibrated, overrides ADC overflow */
    spUncalData = 4,
    /** Returned when the API was unable to keep up with the necessary processing */
    spCPULimited = 5,
    /**
     * This warning is returned when recoverable USB data loss occurred. The measurement
     * may be uncalibrated due to the data loss and may need to be ignored. The device
     * can continue to be used without any action/intervention. This warning can appear
     * in periods of high CPU or USB load.
     */
    spInvalidCalData = 6
};

/**
 * Boolean type. Used in public facing functions instead of `bool` to improve
 * API use from different programming languages.
 */
enum SpBool {
    /** False */
    spFalse = 0,
    /** True */
    spTrue = 1
};

/**
 * Specifies device power state. See @ref powerStates for more information.
 */
enum SpPowerState {
    /** On */
    spPowerStateOn = 0,
    /** Standby */
    spPowerStateStandby = 1
};

/**
 * Measurement mode
 */
enum SpMode {
    /** Idle, no measurement active */
    spModeIdle = 0,
    /** Swept spectrum analysis */
    spModeSweeping = 1,
    /** Real-time spectrum analysis */
    spModeRealTime = 2,
    /** I/Q streaming */
    spModeIQStreaming = 3,
    /** I/Q sweep list / frequency hopping */
    spModeIQSweepList = 4,
    /** Audio demod */
    spModeAudio = 5,
};

/**
 * Detector used for sweep and real-time spectrum analysis.
 */
enum SpDetector {
    /** Average */
    spDetectorAverage = 0,
    /** Min/Max */
    spDetectorMinMax = 1
};

/**
 * Specifies units of sweep and real-time spectrum analysis measurements.
 */
enum SpScale {
    /** dBm */
    spScaleLog = 0,
    /** mV */
    spScaleLin = 1,
    /** Log scale, no corrections */
    spScaleFullScale = 2
};

/**
 * Specifies units in which VBW processing occurs for swept analysis.
 */
enum SpVideoUnits {
    /** dBm */
    spVideoLog = 0,
    /** Linear voltage */
    spVideoVoltage = 1,
    /** Linear power */
    spVideoPower = 2,
    /** No VBW processing */
    spVideoSample = 3
};

/**
 * Specifies the window used for sweep and real-time analysis.
 */
enum SpWindowType {
    /** SRS flattop */
    spWindowFlatTop = 0,
    /** Nutall */
    spWindowNutall = 1,
    /** Blackman */
    spWindowBlackman = 2,
    /** Hamming */
    spWindowHamming = 3,
    /** Gaussian 6dB BW window for EMC measurements and CISPR compatibility */
    spWindowGaussian6dB = 4,
    /** Rectangular (no) window */
    spWindowRect = 5
};

/**
 * Specifies the data type of I/Q data returned from the API.
 * For I/Q streaming and I/Q sweep lists.
 * See @ref iqDataTypes
 */
enum SpDataType {
    /** 32-bit complex floats */
    spDataType32fc = 0,
    /** 16-bit complex shorts */
    spDataType16sc = 1
};

/**
 * External trigger edge polarity for I/Q streaming.
 */
enum SpTriggerEdge {
    /** Rising edge */
    spTriggerEdgeRising = 0,
    /** Falling edge */
    spTriggerEdgeFalling = 1
};

/**
 * Internal GPS state
 */
enum SpGPSState {
    /** GPS is not locked */
    spGPSStateNotPresent = 0,
    /** GPS is locked, NMEA data is valid, but the timebase is not being disciplined by the GPS */
    spGPSStateLocked = 1,
    /** GPS is locked, NMEA data is valid, timebase is being disciplined by the GPS */
    spGPSStateDisciplined = 2
};

/**
 * Used to indicate the source of the timebase reference for the device.
 */
enum SpReference {
    /** Use the internal 10MHz timebase. */
    spReferenceUseInternal = 0,
    /** Use an external 10MHz timebase on the `10 MHz In` port. */
    spReferenceUseExternal = 1
};

/**
 * Used to specify the function of the GPIO port. See #spSetGPIOPort.
 */
enum SpGPIOFunction {
    /**
     * The port will be used for trigger detection while I/Q streaming.
     * The port is not configured until the next I/Q stream is started.
     * This is the default GPIO function.
     */
    SpGPIOFunctionTrigIn = 0,
    /** The internal PPS signal is routed to the GPIO port. */
    SpGPIOFunctionPPSOut = 1,
    /** Set to logic low. */
    SpGPIOFunctionLogicOutLow = 2,
    /** Set to logic high. */
    SpGPIOFunctionLogicOutHigh = 3,
    /** The port can be used to manually write UART messages. */
    SpGPIOFunctionUARTDirect = 4,
    /** The port will be used for frequency switching during sweep. */
    SpGPIOFunctionUARTSweep = 5,
    /** (Not implemented) The port will be used for switching while I/Q streaming. */
    SpGPIOFunctionUARTDoppler = 6
};

/**
 * Audio demodulation type.
 */
enum SpAudioType {
    /** AM */
    spAudioTypeAM = 0,
    /** FM */
    spAudioTypeFM = 1,
    /** Upper side band */
    spAudioTypeUSB = 2,
    /** Lower side band */
    spAudioTypeLSB = 3,
    /** CW */
    spAudioTypeCW = 4
};

class sp_api
{
     /** Used for boolean true when integer parameters are being used. Also see #SpBool. */
    public static int SP_TRUE = 1;
    /** Used for boolean false when integer parameters are being used. Also see #SpBool. */
    public static int SP_FALSE = 0;

    /** Max number of devices that can be interfaced in the API. */
    public static int SP_MAX_DEVICES = 9;

    /** Maximum reference level in dBm */
    public static double SP_MAX_REF_LEVEL = 20.0;
    /** Tells the API to automatically choose attenuation based on reference level. */
    public static int SP_AUTO_ATTEN = -1;
    /** Valid atten values [0,6] or -1 for auto */
    public static int SP_MAX_ATTEN = 6;

    /** Min frequency for sweeps, and min center frequency for I/Q measurements. */
    public static double SP_MIN_FREQ = 9.0e3;
    /** Max frequency for sweeps, and max center frequency for I/Q measurements. */
    public static double SP_MAX_FREQ = 15.0e9;

    /** Min sweep time in seconds. See #spSetSweepCoupling. */
    public static double SP_MIN_SWEEP_TIME = 1.0e-6;
    /** Max sweep time in seconds. See #spSetSweepCoupling. */
    public static double SP_MAX_SWEEP_TIME = 100.0;

    /** Min span for device configured in real-time measurement mode */
    public static double SP_REAL_TIME_MIN_SPAN = 200.0e3;
    /** Max span for device configured in real-time measurement mode */
    public static double SP_REAL_TIME_MAX_SPAN = 40.0e6;
    /** Min RBW for device configured in real-time measurement mode */
    public static double SP_REAL_TIME_MIN_RBW = 2.0e3;
    /** Max RBW for device configured in real-time measurement mode */
    public static double SP_REAL_TIME_MAX_RBW = 1.0e6;

    /** Max decimation for I/Q streaming. */
    public static int SP_MAX_IQ_DECIMATION = 8192;

    /** Maximum number of definable steps in a I/Q sweep */
    public static int SP_MAX_IQ_SWEEP_STEPS = 1000;

    /** Minimum fan set point in Celcius */
    public static double SP_MIN_FAN_SET_POINT = 0.0;
    /** Maximum fan set point in Celcius */
    public static double SP_MAX_FAN_SET_POINT = 60.0;

    /**
    * Maximum number of I/Q sweeps that can be queued up.
    * Valid sweep indices between [0,15].
    * See @ref iqSweepList
    */
    public static int SP_MAX_SWEEP_QUEUE_SZ = 16;

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetDeviceList(int[] serials, ref int deviceCount);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spOpenDevice(ref int device);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spOpenDeviceBySerial(ref int device, int serialNumber);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spCloseDevice(int device);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spPresetDevice(int device);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetPowerState(int device, SpPowerState powerState);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetPowerState(int device, ref SpPowerState powerState);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetSerialNumber(int device, ref int serialNumber);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetFirmwareVersion(int device, ref int major, ref int minor, ref int revision);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetDeviceDiagnostics(int device, ref float voltage, ref float current, ref float temperature);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetCalDate(int device, ref uint lastCalDate);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetReference(int device, SpReference reference);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetReference(int device, ref SpReference reference);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetGPIOPort(int device, SpGPIOFunction func);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetGPIOPort(int device, ref SpGPIOFunction func);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetUARTBaudRate(int device, float rate);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetUARTBaudRate(int device, ref float rate);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spWriteUARTDirect(int device, byte data);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetGPSTimebaseUpdate(int device, SpBool enabled);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetGPSTimebaseUpdate(int device, ref SpBool enabled);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetGPSHoldoverInfo(int device, ref SpBool usingGPSHoldover, ref uint lastHoldoverTime);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetGPSState(int device, ref SpGPSState GPSState);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetRefLevel(int device, double refLevel);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetRefLevel(int device, ref double refLevel);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetAttenuator(int device, int atten);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetAttenuator(int device, ref int atten);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetSweepCenterSpan(int device, double centerFreqHz, double spanHz);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetSweepStartStop(int device, double startFreqHz, double stopFreqHz);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetSweepCoupling(int device, double rbw, double vbw, double sweepTime);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetSweepDetector(int device, SpDetector detector, SpVideoUnits videoUnits);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetSweepScale(int device, SpScale scale);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetSweepWindow(int device, SpWindowType window);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetSweepGPIOSwitching(int device, double[] freqs, byte[] data, int count);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetSweepGPIOSwitchingDisabled(int device);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetRealTimeCenterSpan(int device, double centerFreqHz, double spanHz);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetRealTimeRBW(int device, double rbw);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetRealTimeDetector(int device, SpDetector detector);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetRealTimeScale(int device, SpScale scale, double frameRef, double frameScale);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetRealTimeWindow(int device, SpWindowType window);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQDataType(int device, SpDataType dataType);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQCenterFreq(int device, double centerFreqHz);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetIQCenterFreq(int device, ref double centerFreqHz);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQSampleRate(int device, int decimation);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQSoftwareFilter(int device, SpBool enabled);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQBandwidth(int device, double bandwidth);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQExtTriggerEdge(int device, SpTriggerEdge edge);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQTriggerSentinel(double sentinelValue);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQQueueSize(int device, int units);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQSweepListDataType(int device, SpDataType dataType);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQSweepListCorrected(int device, SpBool corrected);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQSweepListSteps(int device, int steps);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetIQSweepListSteps(int device, ref int steps);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQSweepListFreq(int device, int step, double freq);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQSweepListRef(int device, int step, double level);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQSweepListAtten(int device, int step, int atten);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetIQSweepListSampleCount(int device, int step, uint samples);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetAudioCenterFreq(int device, double centerFreqHz);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetAudioType(int device, SpAudioType audioType);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetAudioFilters(int device,
                                                    double ifBandwidth,
                                                    double audioLpf,
                                                    double audioHpf);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetAudioFMDeemphasis(int device, double deemphasis);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spConfigure(int device, SpMode mode);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetCurrentMode(int device, ref SpMode mode);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spAbort(int device);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetSweepParameters(int device, ref double actualRBW, ref double actualVBW,
                                                       ref double actualStartFreq, ref double binSize, ref int sweepSize);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetRealTimeParameters(int device, ref double actualRBW, ref int sweepSize, ref double actualStartFreq,
                                                          ref double binSize, ref int frameWidth, ref int frameHeight, ref double poi);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetIQParameters(int device, ref double sampleRate, ref double bandwidth);


    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetIQCorrection(int device, ref float scale);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spIQSweepListGetCorrections(int device, float[] corrections);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetSweep(int device, float[] sweepMin, float[] sweepMax, ref long nsSinceEpoch);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetRealTimeFrame(int device, float[] colorFrame, float[] alphaFrame, float[] sweepMin,
                                                     float[] sweepMax, ref int frameCount, ref long nsSinceEpoch);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetIQ(int device, float[] iqBuf, int iqBufSize, double[] triggers, int triggerBufSize,
                                          ref long nsSinceEpoch, SpBool purge, ref int sampleLoss, ref int samplesRemaining);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spIQSweepListGetSweep(int device, float[] dst, long[] timestamps);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spIQSweepListStartSweep(int device, int pos, float[] dst, long[] timestamps);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spIQSweepListFinishSweep(int device, int pos);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetAudio(int device, float[] audio);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetGPSInfo(int device, SpBool refresh, ref SpBool updated, ref long secSinceEpoch,
                                               ref double latitude, ref double longitude, ref double altitude, char[] nmea, ref int nmeaLen);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spWriteToGPS(int device, in byte[] mem, int len);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spSetFanSetPoint(int device, float setPoint);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SpStatus spGetFanSettings(int device, ref float setPoint, ref float voltage);

    public static string spGetAPIString()
    {
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(spGetAPIVersion());
    }

    public static string spGetStatusString(SpStatus status)
    {
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(spGetErrorString(status));
    }

    // Call string variants above instead
    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr spGetErrorString(SpStatus status);

    [DllImport("sp_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr spGetAPIVersion();
}
