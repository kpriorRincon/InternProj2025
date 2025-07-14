using System;
using System.Runtime.InteropServices;

/*
 * The C# API class for the BB series devices is a class of static members and methods which
 * are simply a 1-to-1 mapping of the C API. This makes is easy to modify and look up
 * functions in the API manual.
 */

enum bbStatus
{
    // Configuration Errors
    bbInvalidModeErr             = -112,
    bbReferenceLevelErr          = -111,
    bbInvalidVideoUnitsErr       = -110,
    bbInvalidWindowErr           = -109,
    bbInvalidBandwidthTypeErr    = -108,
    bbInvalidSweepTimeErr        = -107,
    bbBandwidthErr               = -106,
    bbInvalidGainErr             = -105,
    bbAttenuationErr             = -104,
    bbFrequencyRangeErr          = -103,
    bbInvalidSpanErr             = -102,
    bbInvalidScaleErr            = -101,
    bbInvalidDetectorErr         = -100,

    bbInvalidFileSizeErr         = -19,
    bbLibusbError                = -18,
    bbNotSupportedErr            = -17,
    bbTrackingGeneratorNotFound  = -16,

    bbUSBTimeoutErr              = -15,
    bbDeviceConnectionErr        = -14,
    bbPacketFramingErr           = -13,
    bbGPSErr                     = -12,
    bbGainNotSetErr              = -11,
    bbDeviceNotIdleErr           = -10,
    bbDeviceInvalidErr           = -9,
    bbBufferTooSmallErr          = -8,
    bbNullPtrErr                 = -7,
    bbAllocationLimitErr         = -6,
    bbDeviceAlreadyStreamingErr  = -5,
    bbInvalidParameterErr        = -4,
    bbDeviceNotConfiguredErr     = -3,
    bbDeviceNotStreamingErr      = -2,
    bbDeviceNotOpenErr           = -1,

    bbNoError                    = 0,

    // Warnings/Messages
    bbAdjustedParameter          = 1,
    bbADCOverflow                = 2,
    bbNoTriggerFound             = 3,
    bbClampedToUpperLimit        = 4,
    bbClampedToLowerLimit        = 5,
    bbUncalibratedDevice         = 6,
    bbDataBreak                  = 7,
    bbUncalSweep                 = 8,
    bbInvalidCalData             = 9
};

enum bbDataType
{
    bbDataType32fc = 0,
    bbDataType16sc = 1
};

enum bbPowerState
{
    bbPowerStateOn = 0,
    bbPowerStateStandby = 1
};

class bb_api
{
    public static int BB_TRUE = 1;
    public static int BB_FALSE = 0;

    // bbGetDeviceType: type
    public static int BB_DEVICE_NONE = 0;
    public static int BB_DEVICE_BB60A = 1;
    public static int BB_DEVICE_BB60C = 2;
    public static int BB_DEVICE_BB60D = 3;

    public static int BB_MAX_DEVICES = 8;

    // Frequencies specified in Hz
    public static double BB_MIN_FREQ = 9.0e3;
    public static double BB_MAX_FREQ = 6.4e9;

    public static double BB_MIN_SPAN = 20.0;
    public static double BB_MAX_SPAN = BB_MAX_FREQ - BB_MIN_FREQ;

    public static double BB_MIN_RBW = 0.602006912;
    public static double BB_MAX_RBW = 10100000.0;

    public static double BB_MIN_SWEEP_TIME = 0.00001; // 10us
    public static double BB_MAX_SWEEP_TIME = 1.0; // 1s

    // Real-Time
    public static double BB_MIN_RT_RBW = 2465.820313;
    public static double BB_MAX_RT_RBW = 631250.0;
    public static double BB_MIN_RT_SPAN = 200.0e3;
    public static double BB60A_MAX_RT_SPAN = 20.0e6;
    // BB60C/D
    public static double BB60C_MAX_RT_SPAN = 27.0e6;

    public static double BB_MIN_USB_VOLTAGE = 4.4;

    public static double BB_MAX_REFERENCE = 20.0; // dBM

    // Gain/Atten can be integers between -1 and MAX
    public static int BB_AUTO_ATTEN = -1;
    public static int BB_MAX_ATTEN = 3;
    public static int BB_AUTO_GAIN = -1;
    public static int BB_MAX_GAIN = 3;

    public static int BB_MIN_DECIMATION = 1; // No decimation
    public static int BB_MAX_DECIMATION = 8192;

    // bbInitiate: mode
    public static int BB_IDLE = -1;
    public static uint BB_SWEEPING = 0;
    public static uint BB_REAL_TIME = 1;
    public static uint BB_STREAMING = 4;
    public static uint BB_AUDIO_DEMOD = 7;
    public static uint BB_TG_SWEEPING = 8;

    // bbConfigureSweepCoupling: rejection
    public static uint BB_NO_SPUR_REJECT = 0;
    public static uint BB_SPUR_REJECT = 1;

    // bbConfigureAcquisition: scale
    public static uint BB_LOG_SCALE = 0;
    public static uint BB_LIN_SCALE = 1;
    public static uint BB_LOG_FULL_SCALE = 2;
    public static uint BB_LIN_FULL_SCALE = 3;

    // bbConfigureSweepCoupling: rbwShape
    public static uint BB_RBW_SHAPE_NUTTALL = 0;
    public static uint BB_RBW_SHAPE_FLATTOP = 1;
    public static uint BB_RBW_SHAPE_CISPR = 2;

    // bbConfigureAcquisition: detector
    public static uint BB_MIN_AND_MAX = 0;
    public static uint BB_AVERAGE = 1;

    // bbConfigureProcUnits: units
    public static uint BB_LOG = 0;
    public static uint BB_VOLTAGE = 1;
    public static uint BB_POWER = 2;
    public static uint BB_SAMPLE = 3;

    // bbConfigureDemod: modulationType
    public static int BB_DEMOD_AM = 0;
    public static int BB_DEMOD_FM = 1;
    public static int BB_DEMOD_USB = 2;
    public static int BB_DEMOD_LSB = 3;
    public static int BB_DEMOD_CW = 4;

    // Streaming flags
    public static uint BB_STREAM_IQ = 0x0;
    public static uint BB_DIRECT_RF = 0x2; // BB60C/D only
    public static uint BB_TIME_STAMP = 0x10;

    // BB60C/A bbConfigureIO, port1
    public static uint BB60C_PORT1_AC_COUPLED = 0x00;
    public static uint BB60C_PORT1_DC_COUPLED = 0x04;
    public static uint BB60C_PORT1_10MHZ_USE_INT = 0x00;
    public static uint BB60C_PORT1_10MHZ_REF_OUT = 0x100;
    public static uint BB60C_PORT1_10MHZ_REF_IN = 0x8;
    public static uint BB60C_PORT1_OUT_LOGIC_LOW = 0x14;
    public static uint BB60C_PORT1_OUT_LOGIC_HIGH = 0x1C;
    // BB60C/A bbConfigureIO, port2
    public static uint BB60C_PORT2_OUT_LOGIC_LOW = 0x00;
    public static uint BB60C_PORT2_OUT_LOGIC_HIGH = 0x20;
    public static uint BB60C_PORT2_IN_TRIG_RISING_EDGE = 0x40;
    public static uint BB60C_PORT2_IN_TRIG_FALLING_EDGE = 0x60;

    // BB60D bbConfigureIO, port1
    public static uint BB60D_PORT1_DISABLED = 0;
    public static uint BB60D_PORT1_10MHZ_REF_IN = 1;
    // BB60D bbConfigureIO, port2
    public static uint BB60D_PORT2_DISABLED = 0;
    public static uint BB60D_PORT2_10MHZ_REF_OUT = 1;
    public static uint BB60D_PORT2_IN_TRIG_RISING_EDGE = 2;
    public static uint BB60D_PORT2_IN_TRIG_FALLING_EDGE = 3;
    public static uint BB60D_PORT2_OUT_LOGIC_LOW = 4;
    public static uint BB60D_PORT2_OUT_LOGIC_HIGH = 5;
    public static uint BB60D_PORT2_OUT_UART = 6;

    // BB60D only
    // bbSetUARTRate, rate
    public static int BB60D_UART_BAUD_4_8K = 0;
    public static int BB60D_UART_BAUD_9_6K = 1;
    public static int BB60D_UART_BAUD_19_2K = 2;
    public static int BB60D_UART_BAUD_38_4K = 3;
    public static int BB60D_UART_BAUD_14_4K = 4;
    public static int BB60D_UART_BAUD_28_8K = 5;
    public static int BB60D_UART_BAUD_57_6K = 6;
    public static int BB60D_UART_BAUD_115_2K = 7;
    public static int BB60D_UART_BAUD_125K = 8;
    public static int BB60D_UART_BAUD_250K = 9;
    public static int BB60D_UART_BAUD_500K = 10;
    public static int BB60D_UART_BAUD_1000K = 11;

    // For sweep antenna switching and pseudo-doppler
    public static int BB60D_MIN_UART_STATES = 2;
    public static int BB60D_MAX_UART_STATES = 8;

    // bbStoreTgThru: flag
    public static int TG_THRU_0DB = 0x1;
    public static int TG_THRU_20DB = 0x2;

    // bbSetTgReference: reference
    public static int TG_REF_UNUSED = 0;
    public static int TG_REF_INTERNAL_OUT = 1;
    public static int TG_REF_EXTERNAL_IN = 2; // TG124 only

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetSerialNumberList(int[] serialNumbers, ref int deviceCount);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetSerialNumberList2(int[] serialNumbers,
        int[] deviceTypes, ref int deviceCount);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbOpenDevice(ref int device);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbOpenDeviceBySerialNumber(ref int device, int serialNumber);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbCloseDevice(int device);

    // Power state functions BB60D only
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbSetPowerState(int device, bbPowerState powerState);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetPowerState(int device, ref bbPowerState powerState);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbPreset(int device);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbPresetFull(ref int device);

    // Self cal function BB60A only
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbSelfCal(int device);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetSerialNumber(int device, ref uint serialNumber);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetDeviceType(int device, ref int deviceType);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetFirmwareVersion(int device, ref int version);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetDeviceDiagnostics(int device,
        ref float temperature, ref float usbVoltage, ref float usbCurrent);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureIO(int device, uint port1, uint port2);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbSyncCPUtoGPS(int comPort, int baudRate);

    // UART functions BB60D only
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbSetUARTRate(int device, int rate);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbEnableUARTSweeping(int device, in double[] freqs, in byte[] data, int states);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbDisableUARTSweeping(int device);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbEnableUARTStreaming(int device, in byte[] data, in uint[] counts, int states);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbDisableUARTStreaming(int device);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbWriteUARTImm(int device, byte data);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureRefLevel(int device, double refLevel);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureGainAtten(int device, int gain, int atten);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureCenterSpan(int device, double center, double span);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureSweepCoupling(int device,
        double rbw, double vbw, double sweepTime, uint rbwShape, uint rejection);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureAcquisition(int device, uint detector, uint scale);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureProcUnits(int device, uint units);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureRealTime(int device, double frameScale, int frameRate);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureRealTimeOverlap(int device, double advanceRate);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureIQCenter(int device, double centerFreq);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureIQ(int device, int downsampleFactor, double bandwidth);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureIQDataType(int device, bbDataType dataType);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureIQTriggerSentinel(int device, int sentinel);

    // Audio demod
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureDemod(int device, int modulationType, double freq,
        float IFBW, float audioLowPassFreq, float audioHighPassFreq, float FMDeemphasis);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbInitiate(int device, uint mode, uint flag);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbAbort(int device);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbQueryTraceInfo(int device,
        ref uint traceLen, ref double binSize, ref double start);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbQueryRealTimeInfo(int device,
        ref int frameWidth, ref int frameHeight);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbQueryRealTimePoi(int device, ref double poi);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbQueryIQParameters(int device,
        ref double sampleRate, ref double bandwidth);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetIQCorrection(int device, ref float correction);

    [DllImport("bb_api", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbFetchTrace_32f(int device, int arraySize,
        float[] traceMin, float[] traceMax);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbFetchTrace(int device, int arraysize,
        double[] traceMin, double[] traceMax);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbFetchRealTimeFrame(int device,
        float[] traceMin, float[] traceMax, float[] frame, float[] alphaFrame);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetIQUnpacked(int device, float[] iqData, int iqCount,
        int[] triggers, int triggerCount, int purge,
        ref int dataRemaining, ref int sampleLoss, ref int sec, ref int nano);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbFetchAudio(int device, float[] audio);

    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbAttachTg(int device);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbIsTgAttached(int device, ref bool attached);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigTgSweep(int device, int sweepSize,
        bool highDynamicRange, bool passiveDevice);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbStoreTgThru(int device, int flag);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbSetTg(int device, double frequency, double amplitude);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbGetTgFreqAmpl(int device,
        ref double frequency, ref double amplitude);
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbSetTgReference(int device, int reference);

    public static string bbGetDeviceName(int device)
    {
        int device_type = -1;
        bbGetDeviceType(device, ref device_type);
        if (device_type == BB_DEVICE_BB60A)
            return "BB60A";
        if (device_type == BB_DEVICE_BB60C)
            return "BB60C";
        if (device_type == BB_DEVICE_BB60D)
            return "BB60D";

        return "Unknown device";
    }

    public static string bbGetSerialString(int device)
    {
        uint serial_number = 0;
        if (bbGetSerialNumber(device, ref serial_number) == bbStatus.bbNoError)
            return serial_number.ToString();

        return "";
    }

    public static string bbGetFirmwareString(int device)
    {
        int firmware_version = 0;
        if (bbGetFirmwareVersion(device, ref firmware_version) == bbStatus.bbNoError)
            return firmware_version.ToString();

        return "";
    }

    public static string bbGetAPIString()
    {
        IntPtr str_ptr = bbGetAPIVersion();
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(str_ptr);
    }

    public static string bbGetStatusString(bbStatus status)
    {
        IntPtr str_ptr = bbGetErrorString(status);
        return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(str_ptr);
    }

    // Call get_string variants above instead
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr bbGetAPIVersion();
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr bbGetErrorString(bbStatus status);

    // Deprecated functions, use suggested alternatives

    // Use bbConfigureRefLevel instead
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureLevel(int device,
        double refLevel, double atten);
    // Use bbConfigureGainAtten instead
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbConfigureGain(int device, int gain);
    // Use bbQueryIQParameters instead
    [DllImport("bb_api.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bbStatus bbQueryStreamInfo(int device,
        ref int return_len, ref double bandwidth, ref int samples_per_sec);

    // Deprecated macros, use alternatives where available
    public static double BB60_MIN_FREQ = BB_MIN_FREQ;
    public static double BB60_MAX_FREQ = BB_MAX_FREQ;
    public static double BB60_MAX_SPAN = BB_MAX_SPAN;
    public static double BB_MIN_BW = BB_MIN_RBW;
    public static double BB_MAX_BW = BB_MAX_RBW;
    public static double BB_MAX_ATTENUATION = 30.0; // For deprecated bbConfigureLevel function
    public static int BB60C_MAX_GAIN = BB_MAX_GAIN;
    public static uint BB_PORT1_INT_REF_OUT = 0x00;
    public static uint BB_PORT1_EXT_REF_IN = BB60C_PORT1_10MHZ_REF_IN;
    public static uint BB_RAW_PIPE = BB_STREAMING;
    public static uint BB_STREAM_IF = 0x1; // No longer supported
    // Use new device specific port 1 macros
    public static uint BB_PORT1_AC_COUPLED = BB60C_PORT1_AC_COUPLED;
    public static uint BB_PORT1_DC_COUPLED = BB60C_PORT1_DC_COUPLED;
    public static uint BB_PORT1_10MHZ_USE_INT = BB60C_PORT1_10MHZ_USE_INT;
    public static uint BB_PORT1_10MHZ_REF_OUT = BB60C_PORT1_10MHZ_REF_OUT;
    public static uint BB_PORT1_10MHZ_REF_IN = BB60C_PORT1_10MHZ_REF_IN;
    public static uint BB_PORT1_OUT_LOGIC_LOW = BB60C_PORT1_OUT_LOGIC_LOW;
    public static uint BB_PORT1_OUT_LOGIC_HIGH = BB60C_PORT1_OUT_LOGIC_HIGH;
    // Use new device specific port 2 macros
    public static uint BB_PORT2_OUT_LOGIC_LOW = BB60C_PORT2_OUT_LOGIC_LOW;
    public static uint BB_PORT2_OUT_LOGIC_HIGH = BB60C_PORT2_OUT_LOGIC_HIGH;
    public static uint BB_PORT2_IN_TRIGGER_RISING_EDGE = BB60C_PORT2_IN_TRIG_RISING_EDGE;
    public static uint BB_PORT2_IN_TRIGGER_FALLING_EDGE = BB60C_PORT2_IN_TRIG_FALLING_EDGE;
}
