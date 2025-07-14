using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class sp_example
{
    // Small function which illustrates using the C# API to
    // 1) Retrieve diagnostic data from the SP145
    // 2) Configure and retrieve a sweep
    // 3) Configure and retrieve a block of IQ data
    public void Run()
    {
        // Get serial list and open first device
        int handle = -1;
        int[] serials = new int[8];
        int deviceCount = 8;
        SpStatus status = SpStatus.spNoError;        

        status = sp_api.spGetDeviceList(serials, ref deviceCount);
        if (deviceCount < 1)
        {
            Console.WriteLine("No devices found");
            return;
        }

        status = sp_api.spOpenDeviceBySerial(ref handle, serials[0]);

        if (status < 0)
        {
            Console.WriteLine("Error opening device: " + sp_api.spGetStatusString(status));
            return;
        }

        // Print off information about device
        int serialNumber = 0;
        sp_api.spGetSerialNumber(handle, ref serialNumber);
        Console.WriteLine("Serial: " + serialNumber);

        int major = 0, minor = 0, revision = 0;
        sp_api.spGetFirmwareVersion(handle, ref major, ref minor, ref revision);
        Console.WriteLine("Firmware Version: " + major + "." + minor + "." + revision);

        float voltage = 0, current = 0, temperature = 0;
        sp_api.spGetDeviceDiagnostics(handle, ref voltage, ref current, ref temperature);
        Console.WriteLine("Voltage: " + voltage + " V");
        Console.WriteLine("Current: " + current + " A");
        Console.WriteLine("Temperature: " + temperature + " C");

        // Configure leveling
        sp_api.spSetAttenuator(handle, sp_api.SP_AUTO_ATTEN);
        sp_api.spSetRefLevel(handle, 0.0);

        // Setup sweep
        sp_api.spSetSweepCenterSpan(handle, 1.0e9, 20.0e6);
        sp_api.spSetSweepCoupling(handle, 10.0e3, 10.0e3, 0.001);
        sp_api.spSetSweepDetector(handle, SpDetector.spDetectorAverage, SpVideoUnits.spVideoPower);
        sp_api.spSetSweepScale(handle, SpScale.spScaleLog);
        sp_api.spSetSweepWindow(handle, SpWindowType.spWindowFlatTop);

        status = sp_api.spConfigure(handle, SpMode.spModeSweeping);
        if (status < 0)
        {
            Console.WriteLine("spConfigure error: " + sp_api.spGetStatusString(status));
            return;
        }

        double actualRBW = 0, actualVBW = 0, actualStartFreq = 0, binSize = 0;
        int sweepLength = 0;
        sp_api.spGetSweepParameters(handle, ref actualRBW, ref actualVBW, ref actualStartFreq,
            ref binSize, ref sweepLength);

        float[] sweepMin = new float[sweepLength];
        float[] sweepMax = new float[sweepLength];
        long sweepTimeNs = 0;

        sp_api.spGetSweep(handle, sweepMin, sweepMax, ref sweepTimeNs);
        sp_api.spAbort(handle);

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
        sp_api.spSetIQCenterFreq(handle, 1.0e9);
        sp_api.spSetIQSampleRate(handle, 1);
        sp_api.spSetIQBandwidth(handle, 40.0e6);
        sp_api.spSetIQExtTriggerEdge(handle, SpTriggerEdge.spTriggerEdgeRising);

        status = sp_api.spConfigure(handle, SpMode.spModeIQStreaming);
        if (status < 0)
        {
            Console.WriteLine("spConfigure error: " + sp_api.spGetStatusString(status));
            return;
        }

        double iqSampleRate = 0, iqBandwidth = 0;
        sp_api.spGetIQParameters(handle, ref iqSampleRate, ref iqBandwidth);
        Console.WriteLine("IQ Sample Rate: " + iqSampleRate);
        Console.WriteLine("IQ Bandwidth: " + iqBandwidth);

        int iqLen = 1 << 20;
        float[] iq = new float[iqLen * 2];
        int triggerCount = 10;
        double[] triggers = new double[triggerCount];
        long iqTimeStamp = 0;
        int sampleLoss = 0;
        int samplesRemaining = 0;

        sp_api.spGetIQ(handle, iq, iqLen, triggers, triggerCount, ref iqTimeStamp,
            SpBool.spTrue, ref sampleLoss, ref samplesRemaining);
        sp_api.spAbort(handle);

        // Calculate average power of IQ data
        double iqAvgPow = 0;
        for (int i = 0; i < iqLen; i++)
        {
            iqAvgPow += iq[i * 2] * iq[i * 2] + iq[i * 2 + 1] * iq[i * 2 + 1];
        }
        iqAvgPow /= iqLen;

        Console.WriteLine("IQ Avg Power: " + 10 * Math.Log10(iqAvgPow));

        sp_api.spCloseDevice(handle);
    }
}
