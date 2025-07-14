using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class sa_example
{
    // Small function which illustrates using the C# API to
    // 1) Retrieve diagnostic data from the SA series device
    // 2) Configure and retrieve a sweep
    // 3) Configure and retrieve a block of IQ data
    public void Run()
    {
        // Get serial list and open first device
        int handle = -1;
        int[] serials = new int[8];
        int deviceCount = 0;

        saStatus status = sa_api.saGetSerialNumberList(serials, ref deviceCount);
        if (deviceCount < 1)
        {
            Console.WriteLine("No devices found");
            return;
        }

        status = sa_api.saOpenDeviceBySerialNumber(ref handle, serials[0]);
        if (status < 0)
        {
            Console.WriteLine("saOpenDevice error: " + sa_api.saGetStatusString(status));
            return;
        }

        // Print off information about device
        saDeviceType deviceType = saDeviceType.saDeviceTypeSA44;
        int serialNumber = 0;
        sa_api.saGetDeviceType(handle, ref deviceType);
        sa_api.saGetSerialNumber(handle, ref serialNumber);
        Console.WriteLine("Device Type: " + deviceType);
        Console.WriteLine("Serial: " + serialNumber);

        float voltage = 0, temperature = 0;
        sa_api.saQueryDiagnostics(handle, ref voltage);
        sa_api.saQueryTemperature(handle, ref temperature);
        Console.WriteLine("Voltage: " + voltage + " V");
        Console.WriteLine("Temperature: " + temperature + " C");

        // Configure leveling
        sa_api.saConfigGainAtten(handle, sa_api.SA_AUTO_ATTEN, sa_api.SA_AUTO_GAIN, true);
        sa_api.saConfigLevel(handle, 0.0);

        // Setup sweep
        sa_api.saConfigCenterSpan(handle, 1.0e9, 20.0e6);
        sa_api.saConfigSweepCoupling(handle, 10.0e3, 10.0e3, false);
        sa_api.saConfigAcquisition(handle, sa_api.SA_AVERAGE, sa_api.SA_LOG_SCALE);
        sa_api.saConfigRBWShape(handle, sa_api.SA_RBW_SHAPE_FLATTOP);

        status = sa_api.saInitiate(handle, sa_api.SA_SWEEPING, 0);
        if (status < 0)
        {
            Console.WriteLine("saInitiate error: " + sa_api.saGetStatusString(status));
            return;
        }

        int sweepLength = 0;
        double startFreq = 0, binSize = 0;
        sa_api.saQuerySweepInfo(handle, ref sweepLength, ref startFreq, ref binSize);

        float[] sweepMin = new float[sweepLength];
        float[] sweepMax = new float[sweepLength];

        sa_api.saGetSweep_32f(handle, sweepMin, sweepMax);
        sa_api.saAbort(handle);

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
        Console.WriteLine("Peak found: " + 1.0e-9 * (startFreq + binSize * pkIndex) +
            "GHZ, " + sweepMax[pkIndex] + "dBm");

        // Now lets setup up an I/Q stream, 1GHz, no software filtering
        sa_api.saConfigCenterSpan(handle, 1.0e9, 20.0e6);
        sa_api.saConfigIQ(handle, 4, 100.0e3);

        status = sa_api.saInitiate(handle, sa_api.SA_IQ, 0);
        if (status < 0)
        {
            Console.WriteLine("saInitiate error: " + sa_api.saGetStatusString(status));
            return;
        }

        int returnLen = 0;
        double iqSampleRate = 0, iqBandwidth = 0;
        sa_api.saQueryStreamInfo(handle, ref returnLen, ref iqBandwidth, ref iqSampleRate);
        Console.WriteLine("IQ Sample Rate: " + iqSampleRate);
        Console.WriteLine("IQ Bandwidth: " + iqBandwidth);

        float[] iq = new float[returnLen * 2];

        sa_api.saGetIQ_32f(handle, iq);
        sa_api.saAbort(handle);

        // Calculate average power of IQ data
        double iqAvgPow = 0;
        for (int i = 0; i < returnLen; i++)
        {
            iqAvgPow += iq[i * 2] * iq[i * 2] + iq[i * 2 + 1] * iq[i * 2 + 1];
        }
        iqAvgPow /= returnLen;

        Console.WriteLine("IQ Avg Power: " + 10 * Math.Log10(iqAvgPow));

        sa_api.saCloseDevice(handle);
    }
}
