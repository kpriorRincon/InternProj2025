using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class sm_example
{
    public static bool IS_NETWORKED = false; // True for SM200C and SM435C

    // Small function which illustrates using the C# API to
    // 1) Retrieve diagnostic data from the SM series analyzer
    // 2) Configure and retrieve a sweep
    // 3) Configure and retrieve a block of IQ data
    public void Run()
    {
        // Get serial list and open first device
        int handle = -1;
        int[] serials = new int[8];
        int deviceCount = 0;
        SmStatus status = SmStatus.smNoError;

        if (IS_NETWORKED)
        {
            status = sm_api.smOpenNetworkedDevice(ref handle, sm_api.SM_ADDR_ANY, sm_api.SM_DEFAULT_ADDR, (ushort)sm_api.SM_DEFAULT_PORT);
        }
        else
        {
            status = sm_api.smGetDeviceList(serials, ref deviceCount);
            if (deviceCount < 1)
            {
                Console.WriteLine("No devices found");
                return;
            }

            status = sm_api.smOpenDeviceBySerial(ref handle, serials[0]);
        }

        if (status < 0)
        {
            Console.WriteLine("Error opening device: " + sm_api.smGetStatusString(status));
            return;
        }

        // Print off information about device
        SmDeviceType deviceType = SmDeviceType.smDeviceTypeSM200A;
        int serialNumber = 0;
        sm_api.smGetDeviceInfo(handle, ref deviceType, ref serialNumber);
        Console.WriteLine("Device Type: " + deviceType);
        Console.WriteLine("Serial: " + serialNumber);

        int major = 0, minor = 0, revision = 0;
        float voltage = 0, current = 0, temperature = 0;
        sm_api.smGetFirmwareVersion(handle, ref major, ref minor, ref revision);
        Console.WriteLine("Firmware Version: " + major + "." + minor + "." + revision);

        sm_api.smGetDeviceDiagnostics(handle, ref voltage, ref current, ref temperature);
        Console.WriteLine("Voltage: " + voltage + " V");
        Console.WriteLine("Current: " + current + " A");
        Console.WriteLine("Temperature: " + temperature + " C");

        // Configure leveling
        sm_api.smSetAttenuator(handle, sm_api.SM_AUTO_ATTEN);
        sm_api.smSetRefLevel(handle, 0.0);

        // Setup sweep
        sm_api.smSetSweepSpeed(handle, SmSweepSpeed.smSweepSpeedAuto);
        sm_api.smSetSweepCenterSpan(handle, 1.0e9, 20.0e6);
        sm_api.smSetSweepCoupling(handle, 10.0e3, 10.0e3, 0.001);
        sm_api.smSetSweepDetector(handle, SmDetector.smDetectorAverage, SmVideoUnits.smVideoPower);
        sm_api.smSetSweepScale(handle, SmScale.smScaleLog);
        sm_api.smSetSweepWindow(handle, SmWindowType.smWindowFlatTop);
        sm_api.smSetSweepSpurReject(handle, SmBool.smFalse);

        status = sm_api.smConfigure(handle, SmMode.smModeSweeping);
        if (status < 0)
        {
            Console.WriteLine("smConfigure error: " + sm_api.smGetStatusString(status));
            return;
        }

        double actualRBW = 0, actualVBW = 0, actualStartFreq = 0, binSize = 0;
        int sweepLength = 0;
        sm_api.smGetSweepParameters(handle, ref actualRBW, ref actualVBW, ref actualStartFreq,
            ref binSize, ref sweepLength);

        float[] sweepMin = new float[sweepLength];
        float[] sweepMax = new float[sweepLength];
        long sweepTimeNs = 0;

        sm_api.smGetSweep(handle, sweepMin, sweepMax, ref sweepTimeNs);
        sm_api.smAbort(handle);

        // Find peak amplitude in sweep
        int pkIndex = 0;
        for (int i = 1; i < sweepLength; i++)
        {
            if (sweepMax[i] > sweepMax[pkIndex])
            {
                pkIndex = i;
            }
        }

        // Print off the peak freq/ampl of the sweep
        Console.WriteLine("Peak found: " + 1.0e-9 * (actualStartFreq + binSize * pkIndex) +
            "GHZ, " + sweepMax[pkIndex] + "dBm");

        // Now lets setup up an I/Q stream, 1GHz, no software filtering
        sm_api.smSetIQCenterFreq(handle, 1.0e9);
        sm_api.smSetIQSampleRate(handle, 1);
        sm_api.smSetIQBandwidth(handle, SmBool.smFalse, 40.0e6);
        sm_api.smSetIQExtTriggerEdge(handle, SmTriggerEdge.smTriggerEdgeRising);

        status = sm_api.smConfigure(handle, SmMode.smModeIQ);
        if (status < 0)
        {
            Console.WriteLine("smConfigure error: " + sm_api.smGetStatusString(status));
            return;
        }

        double iqSampleRate = 0, iqBandwidth = 0;
        sm_api.smGetIQParameters(handle, ref iqSampleRate, ref iqBandwidth);
        Console.WriteLine("IQ Sample Rate: " + iqSampleRate);
        Console.WriteLine("IQ Bandwidth: " + iqBandwidth);

        int iqLen = 1 << 20;
        float[] iq = new float[iqLen * 2];
        int triggerCount = 10;
        double[] triggers = new double[triggerCount];
        long iqTimeStamp = 0;
        int sampleLoss = 0;
        int samplesRemaining = 0;

        sm_api.smGetIQ(handle, iq, iqLen, triggers, triggerCount, ref iqTimeStamp,
            SmBool.smTrue, ref sampleLoss, ref samplesRemaining);
        sm_api.smAbort(handle);

        // Calculate average power of IQ data
        double iqAvgPow = 0;
        for (int i = 0; i < iqLen; i++)
        {
            iqAvgPow += iq[i * 2] * iq[i * 2] + iq[i * 2 + 1] * iq[i * 2 + 1];
        }
        iqAvgPow /= iqLen;

        Console.WriteLine("IQ Avg Power: " + 10 * Math.Log10(iqAvgPow));

        // Speed test for networked devices
        if (IS_NETWORKED)
        {
            double bytesPerSecond = -1;
            sm_api.smNetworkedSpeedTest(handle, 1, ref bytesPerSecond);

            Console.WriteLine("Networked Device Speed: " + bytesPerSecond * 8 / 1e9 + " Gb/s");
        }

        sm_api.smCloseDevice(handle);
    }
}
