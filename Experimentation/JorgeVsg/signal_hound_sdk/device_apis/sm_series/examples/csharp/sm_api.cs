using System;
using System.Runtime.InteropServices;
using System.Text;

/*
 * The C# API class for the SM series devices is a class of static members and methods which
 * are simply a 1-to-1 mapping of the C API. This makes is easy to modify and look up
 * functions in the API manual.
 */

enum SmStatus
{
    smCalErr = -1003, // Internal use
    smMeasErr = -1002, // Internal use
    smErrorIOErr = -1001, // Internal use

    // Calibration file unable to be used with the API
    smInvalidCalibrationFileErr = -200,
    // Invalid center frequency specified
    smInvalidCenterFreqErr = -101,
    // IQ decimation value provided not a valid value
    smInvalidIQDecimationErr = -100,

    smJESDErr = -54,
    // Socket/network error
    smNetworkErr = -53,
    // If the core FX3 program fails to run
    smFx3RunErr = -52,
    // Only can connect up to SM_MAX_DEVICES receivers
    smMaxDevicesConnectedErr = -51,
    // FPGA boot error
    smFPGABootErr = -50,
    // Boot error
    smBootErr = -49,

    // Requesting GPS information when the GPS is not locked
    smGpsNotLockedErr = -16,
    // Invalid API version for target device, TBD
    smVersionMismatchErr = -14,
    // Unable to allocate resources needed to configure the measurement mode
    smAllocationErr = -13,

    // Returned when the device detects framing issue on measurement data
    // Measurement results are likely invalid. Device should be preset/power cycled
    smSyncErr = -11,
    // Invalid or already active sweep position
    smInvalidSweepPosition = -10,
    // Attempting to perform an operation that cannot currently be performed.
    // Often the result of trying to do something while the device is currently
    //   making measurements or not in an idle state.
    smInvalidConfigurationErr = -8,
    // Device disconnected, likely USB error detected
    smConnectionLostErr = -6,
    // Required parameter found to have invalid value
    smInvalidParameterErr = -5,
    // One or more required pointer parameters were null
    smNullPtrErr = -4,
    // User specified invalid device index
    smInvalidDeviceErr = -3,
    // Unable to open device
    smDeviceNotFoundErr = -2,

    // Function returned successfully
    smNoError = 0,

    // One or more of the provided settings were adjusted
    smSettingClamped = 1,
    // Measurement includes data which caused an ADC overload (clipping/compression)
    smAdcOverflow = 2,
    // Measurement is uncalibrated, overrides ADC overflow
    smUncalData = 3,
    // Temperature drift occured, measurements uncalibrated, reconfigure the device
    smTempDriftWarning = 4,
    // Warning when the preselector span is smaller than the user selected span
    smSpanExceedsPreselector = 5,
    // Warning when the internal temperature gets too hot. The device is close to shutting down
    smTempHighWarning = 6,
    // Returned when the API was unable to keep up with the necessary processing
    smCpuLimited = 7,
    // Returned when the API detects a device with newer features than what was available
    //   when this version of the API was released. Suggested fix, update the API.
    smUpdateAPI = 8,
    // Calibration data potentially corrupt
    smInvalidCalData = 9,
};

enum SmDataType
{
    smDataType32fc = 0,
    smDataType16sc = 1
};

enum SmMode
{
    smModeIdle = 0,
    smModeSweeping = 1,
    smModeRealTime = 2,
    smModeIQStreaming = 3,
    smModeIQSegmentedCapture = 5,
    smModeIQSweepList = 6,
    smModeAudio = 4,

    // Deprecated
    smModeIQ = 3 // Use smModeIQStreaming
};

enum SmSweepSpeed
{
    smSweepSpeedAuto = 0,
    smSweepSpeedNormal = 1,
    smSweepSpeedFast = 2
};

enum SmIQStreamSampleRate
{
    smIQStreamSampleRateNative = 0,
    smIQStreamSampleRateLTE = 1
};

enum SmPowerState
{
    smPowerStateOn = 0,
    smPowerStateStandby = 1
};

enum SmDetector
{
    smDetectorAverage = 0,
    smDetectorMinMax = 1
};

enum SmScale
{
    smScaleLog = 0, // Sweep in dBm
    smScaleLin = 1, // Sweep in mV
    smScaleFullScale = 2 // N/A
};

enum SmVideoUnits
{
    smVideoLog = 0,
    smVideoVoltage = 1,
    smVideoPower = 2,
    smVideoSample = 3
};

enum SmWindowType
{
    smWindowFlatTop = 0,
    // 1 (N/A)
    smWindowNutall = 2,
    smWindowBlackman = 3,
    smWindowHamming = 4,
    smWindowGaussian6dB = 5,
    smWindowRect = 6
};

enum SmTriggerType
{
    smTriggerTypeImm = 0,
    smTriggerTypeVideo = 1,
    smTriggerTypeExt = 2,
    smTriggerTypeFMT = 3
};

enum SmTriggerEdge
{
    smTriggerEdgeRising = 0,
    smTriggerEdgeFalling = 1
};

enum SmBool
{
    smFalse = 0,
    smTrue = 1
};

enum SmGPIOState
{
    smGPIOStateOutput = 0,
    smGPIOStateInput = 1
};

enum SmReference
{
    smReferenceUseInternal = 0,
    smReferenceUseExternal = 1
};

enum SmDeviceType
{
    smDeviceTypeSM200A = 0,
    smDeviceTypeSM200B = 1,
    smDeviceTypeSM200C = 2,
    smDeviceTypeSM435B = 3,
    smDeviceTypeSM435C = 4
};

enum SmAudioType
{
    smAudioTypeAM = 0,
    smAudioTypeFM = 1,
    smAudioTypeUSB = 2,
    smAudioTypeLSB = 3,
    smAudioTypeCW = 4
};

enum SmGPSState
{
    smGPSStateNotPresent = 0,
    smGPSStateLocked = 1,
    smGPSStateDisciplined = 2
};

[StructLayout(LayoutKind.Sequential)]
public struct SmGPIOStep
{
    public double freq;
    public Byte mask; // gpio bits

    public SmGPIOStep(double f, Byte m)
    {
        this.freq = f;
        this.mask = m;
    }
}

[StructLayout(LayoutKind.Sequential)]
public struct SmDeviceDiagnostics
{
    float voltage;
    float currentInput;
    float currentOCXO;
    float current58;
    float tempFPGAInternal;
    float tempFPGANear;
    float tempOCXO;
    float tempVCO;
    float tempRFBoardLO;
    float tempPowerSupply;
}

class sm_api
{
    public static int SM_INVALID_HANDLE = -1;

    public static int SM_TRUE = 1;
    public static int SM_FALSE = 0;

    public static int SM_MAX_DEVICES = 9;

    // For networked (10GbE devices)
    public static string SM_ADDR_ANY = "0.0.0.0";
    public static string SM_DEFAULT_ADDR = "192.168.2.10";
    public static int SM_DEFAULT_PORT = 51665;

    public static int SM_AUTO_ATTEN = -1;
    // Valid atten values [0,6] or -1 for auto
    public static int SM_MAX_ATTEN = 6;

    public static double SM_MAX_REF_LEVEL = 20.0;

    // Maximum number of sweeps that can be queued up
    // Sweep indices [0,15]
    public static int SM_MAX_SWEEP_QUEUE_SZ = 16;

    public static double SM200_MIN_FREQ = 100.0e3;
    public static double SM200_MAX_FREQ = 20.6e9;

    public static double SM435_MIN_FREQ = 100.0e3;
    public static double SM435_MAX_FREQ = 44.2e9;
    public static double SM435_MAX_FREQ_IF_OPT = 40.9e9;

    public static int SM_MAX_IQ_DECIMATION = 4096;

    // The frequency at which the manually controlled preselector filters end.
    // Past this frequency, the preselector filters are always enabled.
    public static double SM_PRESELECTOR_MAX_FREQ = 645.0e6;

    // Minimum RBW for fast sweep with Nuttall window
    public static double SM_FAST_SWEEP_MIN_RBW = 30.0e3;

    // Min/max span for device configured in RTSA measurement mode
    public static double SM_REAL_TIME_MIN_SPAN = 200.0e3;
    public static double SM_REAL_TIME_MAX_SPAN = 160.0e6;

    // Sweep time range [1us, 100s]
    public static double SM_MIN_SWEEP_TIME = 1.0e-6;
    public static double SM_MAX_SWEEP_TIME = 100.0;

    // Max number of bytes per SPI transfer
    public static int SM_SPI_MAX_BYTES = 4;

    // For GPIO sweeps
    public static int SM_GPIO_SWEEP_MAX_STEPS = 64;

    // For IQ GPIO switching
    public static int SM_GPIO_SWITCH_MAX_STEPS = 64;
    public static int SM_GPIO_SWITCH_MIN_COUNT = 2;
    public static int SM_GPIO_SWITCH_MAX_COUNT = 4194303 - 1; // 2^22 - 1

    // FPGA internal temperature (Celsius)
    // Returned from smGetDeviceDiagnostics()
    public static double SM_TEMP_WARNING = 95.0;
    public static double SM_TEMP_MAX = 102.0;

    // Segmented I/Q captures, SM200B/SM435B only
    public static int SM_MAX_SEGMENTED_IQ_SEGMENTS = 250;
    public static int SM_MAX_SEGMENTED_IQ_SAMPLES = 520000000;

    // IF output option devices only
    public static double SM435_IF_OUTPUT_FREQ = 1.5e9;
    public static double SM435_IF_OUTPUT_MIN_FREQ = 24.0e9;
    public static double SM435_IF_OUTPUT_MAX_FREQ = 43.5e9;

    // USB devices only
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetDeviceList(int[] serials, ref int deviceCount);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetDeviceList(int[] serials, SmDeviceType[] deviceTypes, ref int deviceCount);

    // USB devices only
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smOpenDevice(ref int device);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smOpenDeviceBySerial(ref int device, int serialNumber);

    // Networked devices only
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smOpenNetworkedDevice(ref int device, string hostAddr, string deviceAddr, ushort port);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smCloseDevice(int device);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smPreset(int device);
    // Preset a device that has not been opened with the smOpenDevice functions
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smPresetSerial(int serialNumber);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smNetworkedSpeedTest(int device, double durationSeconds, ref double bytesPerSecond);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetDeviceInfo(int device,
        ref SmDeviceType deviceType, ref int serialNumber);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetFirmwareVersion(int device,
        ref int major, ref int minor, ref int revision);
    // SM435 only
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smHasIFOutput(int device, ref SmBool present);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetDeviceDiagnostics(int device,
        ref float voltage, ref float current, ref float temperature);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetFullDeviceDiagnostics(int device, ref SmDeviceDiagnostics diagnostics);
    // Networked devices only
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetSFPDiagnostics(int device,
        ref float temp, ref float voltage, ref float txPower, ref float rxPower);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetPowerState(int device, SmPowerState powerState);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetPowerState(int device, ref SmPowerState powerState);

    // Overrides reference level when set to non-auto values
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetAttenuator(int device, int atten);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetAttenuator(int device, ref int atten);

    // Uses this when attenuation is automatic
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetRefLevel(int device, double refLevel);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetRefLevel(int device, ref double refLevel);

    // Set preselector state for all measurement modes
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetPreselector(int device, SmBool enabled);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetPreselector(int device, ref SmBool enabled);

    // Configure IO routines
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetGPIOState(int device,
        SmGPIOState lowerState, SmGPIOState upperState);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetGPIOState(int device,
        ref SmGPIOState lowerState, ref SmGPIOState upperState);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smWriteGPIOImm(int device, byte data);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smReadGPIOImm(int device, ref byte data);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smWriteSPI(int device, uint data, int byteCount);
    // For standard sweeps only
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetGPIOSweepDisabled(int device);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetGPIOSweep(int device, [In, Out] SmGPIOStep[] steps, int stepCount);
    // For IQ streaming only
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetGPIOSwitchingDisabled(int device);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetGPIOSwitching(int device, byte[] gpio,
        uint[] counts, int gpioSteps);

    // Enable the external reference out port
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetExternalReference(int device, SmBool enabled);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetExternalReference(int device, ref SmBool enabled);
    // Specify whether to use the internal reference or reference on the ref in port
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetReference(int device, SmReference reference);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetReference(int device, ref SmReference reference);

    // Enable whether or not the API auto updates the timebase calibration
    // value when a valid GPS lock is acquired.
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetGPSTimebaseUpdate(int device, SmBool enabled);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetGPSTimebaseUpdate(int device, ref SmBool enabled);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetGPSHoldoverInfo(int device,
        ref SmBool usingGPSHoldover, ref ulong lastHoldoverTime);

    // Returns whether the GPS is locked, can be called anytime
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetGPSState(int device, ref SmGPSState GPSState);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSweepSpeed(int device, SmSweepSpeed sweepSpeed);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSweepCenterSpan(int device, double centerFreqHz,
        double spanHz);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSweepStartStop(int device, double startFreqHz,
        double stopFreqHz);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSweepCoupling(int device, double rbw, double vbw,
        double sweepTime);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSweepDetector(int device, SmDetector detector,
        SmVideoUnits videoUnits);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSweepScale(int device, SmScale scale);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSweepWindow(int device, SmWindowType window);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSweepSpurReject(int device, SmBool spurRejectEnabled);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetRealTimeCenterSpan(int device, double centerFreqHz,
        double spanHz);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetRealTimeRBW(int device, double rbw);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetRealTimeDetector(int device, SmDetector detector);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetRealTimeScale(int device, SmScale scale,
        double frameRef, double frameScale);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetRealTimeWindow(int device, SmWindowType window);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQBaseSampleRate(int device, SmIQStreamSampleRate sampleRate);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQDataType(int device, SmDataType dataType);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQCenterFreq(int device, double centerFreqHz);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetIQCenterFreq(int device, ref double centerFreqHz);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQSampleRate(int device, int decimation);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQBandwidth(int device, SmBool enableSoftwareFilter,
        double bandwidth);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQExtTriggerEdge(int device, SmTriggerEdge edge);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQTriggerSentinel(double sentinelValue);
    // Please read the API manual before using this function.
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQQueueSize(int device, float ms);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQSweepListDataType(int device, SmDataType dataType);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQSweepListCorrected(int device, SmBool corrected);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQSweepListSteps(int device, int steps);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetIQSweepListSteps(int device, ref int steps);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQSweepListFreq(int device, int step, double freq);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQSweepListRef(int device, int step, double level);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQSweepListAtten(int device, int step, int atten);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQSweepListSampleCount(int device, int step, uint samples);

    // Segmented I/Q configuration, SM200B/SM435B only
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSegIQDataType(int device, SmDataType dataType);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSegIQCenterFreq(int device, double centerFreqHz);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSegIQVideoTrigger(int device, double triggerLevel, SmTriggerEdge triggerEdge);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSegIQExtTrigger(int device, SmTriggerEdge extTriggerEdge);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSegIQFMTParams(int device, int fftSize, ref double frequencies, ref double ampls, int count);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSegIQSegmentCount(int device, int segmentCount);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetSegIQSegment(int device, int segment, SmTriggerType triggerType,
        int preTrigger, int captureSize, double timeoutSeconds);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetAudioCenterFreq(int device, double centerFreqHz);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetAudioType(int device, SmAudioType audioType);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetAudioFilters(int device, double ifBandwidth,
        double audioLpf, double audioHpf);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetAudioFMDeemphasis(int device, double deemphasis);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smConfigure(int device, SmMode mode);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetCurrentMode(int device, ref SmMode mode);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smAbort(int device);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetSweepParameters(int device, ref double actualRBW,
        ref double actualVBW, ref double actualStartFreq, ref double binSize, ref int sweepSize);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetRealTimeParameters(int device, ref double actualRBW,
        ref int sweepSize, ref double actualStartFreq, ref double binSize, ref int frameWidth,
        ref int frameHeight, ref double poi);
    // Retrieve I/Q parameters for streaming and segmented I/Q captures.
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetIQParameters(int device, ref double sampleRate,
        ref double bandwidth);
    // Retrieve the correction factor/scalar for streaming and segmented I/Q captures.
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetIQCorrection(int device, ref float scale);
    // Retrieve correction factors for an I/Q sweep list measurement.
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smIQSweepListGetCorrections(int device, ref float corrections);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSegIQGetMaxCaptures(int device, ref int maxCaptures);

    // Performs a single sweep, blocking function
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetSweep(int device, float[] sweepMin,
        float[] sweepMax, ref long nsSinceEpoch);

    // Queued sweep mechanisms
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smStartSweep(int device, int pos);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smFinishSweep(int device, int pos,
        float[] sweepMin, float[] sweepMax, ref long nsSinceEpoch);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetRealTimeFrame(int device, float[] colorFrame,
        float[] alphaFrame, float[] sweepMin, float[] sweepMax, ref int frameCount,
        ref long nsSinceEpoch);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetIQ(int device, float[] iqBuf, int iqBufSize,
        double[] triggers, int triggerBufSize, ref long nsSinceEpoch, SmBool purge,
        ref int sampleLoss, ref int samplesRemaining);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smIQSweepListGetSweep(int device, float[] dst, long[] timestamps);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smIQSweepListStartSweep(int device, int pos, float[] dst, long[] timestamps);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smIQSweepListFinishSweep(int device, int pos);

    // Segmented I/Q acquisition functions, SM200B/SM435B only
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSegIQCaptureStart(int device, int capture);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSegIQCaptureWait(int device, int capture);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSegIQCaptureWaitAsync(int device, int capture, ref SmBool completed);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSegIQCaptureTimeout(int device, int capture, int segment, ref SmBool timedOut);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSegIQCaptureTime(int device, int capture, int segment, ref long nsSinceEpoch);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSegIQCaptureRead(int device, int capture, int segment, float[] iq, int offset, int len);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSegIQCaptureFinish(int device, int capture);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSegIQCaptureFull(int device, int capture, float[] iq, int offset, int len,
        ref long nsSinceEpoch, ref SmBool timedOut);
    // Convenience function to resample a 250 MS/s capture to the LTE rate of 245.76 MS/s
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSegIQLTEResample(ref float input, int inputLen, ref float output, ref int outputLen, bool clearDelayLine);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQFullBandAtten(int device, int atten);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQFullBandCorrected(int device, SmBool corrected);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQFullBandSamples(int device, int samples);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQFullBandTriggerType(int device, SmTriggerType triggerType);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQFullBandVideoTrigger(int device, double triggerLevel);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIQFullBandTriggerTimeout(int device, double triggerTimeout);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetIQFullBand(int device, float[] iq, int freq);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetIQFullBandSweep(int device, float[] iq, int startIndex, int stepSize, int steps);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetAudio(int device, float[] audio);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetGPSInfo(int device, SmBool refresh,
        ref SmBool updated, ref long secSinceEpoch, ref double latitude,
        ref double longitude, ref double altitude, char[] nmea, ref int nmeaLen);

    // Device must have GPS write capability. See API manual for more information.
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smWriteToGPS(int device, in byte[] mem, int len);

    // Accepts values between [10-90] as the temp threshold for when the fan turns on
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetFanThreshold(int device, int temp);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetFanThreshold(int device, ref int temp);

    // Must be SM435 device with IF output option
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smSetIFOutput(int device, double frequency);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smGetCalDate(int device, ref ulong lastCalDate);

    // Configure 10GbE network parameters via the network
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smBroadcastNetworkConfig(StringBuilder hostAddr, StringBuilder deviceAddr, ushort port, SmBool nonVolatile);
    // Configure 10GbE network parameters via USB
    // Device handle for these functions cannot be used with others
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smNetworkConfigGetDeviceList(int[] serials, ref int deviceCount);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smNetworkConfigOpenDevice(ref int device, int serialNumber);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smNetworkConfigCloseDevice(int device);    
    public static string smNetworkConfigGetMACString(int device)
    {
        IntPtr strPtr = Marshal.StringToHGlobalAnsi("XX-XX-XX-XX-XX-XX\0");
        smNetworkConfigGetMAC(device, strPtr);
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(strPtr);
    }
    public static SmStatus smNetworkConfigSetIPString(int device, string addr, SmBool nonVolatile)
    {
        IntPtr strPtr = System.Runtime.InteropServices.Marshal.StringToCoTaskMemAnsi(addr);
        return smNetworkConfigSetIP(device, strPtr, nonVolatile);
    }
    public static string smNetworkConfigGetIPString(int device)
    {
        IntPtr strPtr = Marshal.StringToHGlobalAnsi("XXX.XXX.XXX.XXX\0");
        smNetworkConfigGetIP(device, strPtr);
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(strPtr);
    }
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smNetworkConfigSetPort(int device, int port, SmBool nonVolatile);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern SmStatus smNetworkConfigGetPort(int device, ref int port);

    public static string smGetAPIString()
    {
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(smGetAPIVersion());
    }

    public static string smGetStatusString(SmStatus status)
    {
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(smGetErrorString(status));
    }

    public static string smGetProductString()
    {
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(smGetProductID());
    }

    // Call string variants above instead
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern SmStatus smNetworkConfigGetMAC(int device, IntPtr mac);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern SmStatus smNetworkConfigSetIP(int device, in IntPtr addr, SmBool nonVolatile);
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern SmStatus smNetworkConfigGetIP(int device, IntPtr addr);

    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr smGetAPIVersion();
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr smGetProductID();
    [DllImport("sm_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr smGetErrorString(SmStatus status);
}
