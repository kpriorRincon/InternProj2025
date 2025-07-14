using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

class bb_example
{
    // Small function which illustrates using the C# API to
    // 1) Retrieve diagnostic data from the BB series device
    // 2) Configure and retrieve a sweep
    // 3) Configure and retrieve a block of IQ data
    public void Run()
    {
        int id = -1;
        bbStatus status = bbStatus.bbNoError;

        Console.Write("Opening Device, Please Wait\n");
        status = bb_api.bbOpenDevice(ref id);
        if (status != bbStatus.bbNoError)
        {
            Console.Write("Error: Unable to open BB60\n");
            Console.Write(bb_api.bbGetStatusString(status) + "\n");
            promptUserInput();
            return;
        }
        else
        {
            Console.Write("Device Found\n\n");
        }

        Console.Write("API Version: " + bb_api.bbGetAPIString() + "\n");
        Console.Write("Device Type: " + bb_api.bbGetDeviceName(id) + "\n");
        Console.Write("Serial Number: " + bb_api.bbGetSerialString(id) + "\n");
        Console.Write("Firmware Version: " + bb_api.bbGetFirmwareString(id) + "\n");
        Console.Write("\n");

        float temp = 0.0F, voltage = 0.0F, current = 0.0F;
        bb_api.bbGetDeviceDiagnostics(id, ref temp, ref voltage, ref current);
        Console.Write("Device Diagnostics\n" +
            "Temperature: " + temp.ToString() + " C\n" +
            "USB Voltage: " + voltage.ToString() + " V\n" +
            "USB Current: " + current.ToString() + " mA\n");
        Console.Write("\n");

        Console.Write("Configuring Device For a Sweep\n");
        bb_api.bbConfigureAcquisition(id, bb_api.BB_MIN_AND_MAX, bb_api.BB_LOG_SCALE);
        bb_api.bbConfigureCenterSpan(id, 1.0e9, 20.0e6);
        bb_api.bbConfigureRefLevel(id, -20.0);
        bb_api.bbConfigureGainAtten(id, bb_api.BB_AUTO_GAIN, bb_api.BB_AUTO_ATTEN);
        bb_api.bbConfigureSweepCoupling(id, 10.0e3, 10.0e3, 0.001,
            bb_api.BB_RBW_SHAPE_NUTTALL, bb_api.BB_NO_SPUR_REJECT);
        bb_api.bbConfigureProcUnits(id, bb_api.BB_LOG);

        status = bb_api.bbInitiate(id, bb_api.BB_SWEEPING, 0);
        if (status != bbStatus.bbNoError)
        {
            Console.Write("Error: Unable to initialize BB60\n");
            Console.Write(bb_api.bbGetStatusString(status) + "\n");
            promptUserInput();
            return;
        }

        uint traceLen = 0;
        double binSize = 0.0;
        double startFreq = 0.0;
        status = bb_api.bbQueryTraceInfo(id, ref traceLen, ref binSize, ref startFreq);

        float[] traceMax, traceMin;
        traceMax = new float[traceLen];
        traceMin = new float[traceLen];

        bb_api.bbFetchTrace_32f(id, unchecked((int)traceLen), traceMin, traceMax);
        Console.Write("Sweep Retrieved\n\n");

        Console.Write("Configuring the device to stream I/Q data\n");
        bb_api.bbConfigureCenterSpan(id, 1.0e9, 20.0e6);
        bb_api.bbConfigureRefLevel(id, -20.0);
        bb_api.bbConfigureGainAtten(id, bb_api.BB_AUTO_GAIN, bb_api.BB_AUTO_ATTEN);
        bb_api.bbConfigureIQ(id, bb_api.BB_MIN_DECIMATION, 20.0e6);

        status = bb_api.bbInitiate(id, bb_api.BB_STREAMING, bb_api.BB_STREAM_IQ);
        if (status != bbStatus.bbNoError)
        {
            Console.Write("Error: Unable to initialize BB60 for streaming\n");
            Console.Write(bb_api.bbGetStatusString(status) + "\n");
            promptUserInput();
            return;
        }

        int iqCount = 1 << 20;
        int triggerCount = 16;
        double sampleRate = 0.0;
        double bandwidth = 0.0;
        bb_api.bbQueryIQParameters(id, ref sampleRate, ref bandwidth);
        Console.Write("Initialized Stream for \n");
        Console.Write("Samples per second: " + (sampleRate / 1.0e6).ToString() + " MS/s\n");
        Console.Write("Bandwidth: " + (bandwidth/1.0e6).ToString() + " MHz\n");

        // Alternating I/Q samples
        // return_len is the number of I/Q pairs, so.. allocate twice as many floats
        float[] iqData = new float[iqCount * 2];
        int[] triggers = new int[triggerCount];
        int dataRemaining = 0, sampleLoss = 0, iqSec = 0, iqNano = 0;

        bb_api.bbGetIQUnpacked(id, iqData, iqCount, triggers, triggerCount, 1,
            ref dataRemaining, ref sampleLoss, ref iqSec, ref iqNano);
        Console.Write("Retrieved one I/Q packet\n\n");

        Console.Write("Closing Device\n");
        bb_api.bbCloseDevice(id);

        promptUserInput();
    }

    private static void promptUserInput()
    {
        Console.Write("Press enter to end the program\n");
        Console.Read();
    }
}
