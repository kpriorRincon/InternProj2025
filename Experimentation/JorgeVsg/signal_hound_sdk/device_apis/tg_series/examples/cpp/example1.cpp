// Use zero for single device
int device = 0;

// Initialize device
if (tgOpenDevice(device) == tgNoError) {
    // Success
}

// Set frequency to 1 GHz and amplitude to -15 dBm
// Start outputting CW
tgSetFreqAmp(device, 1.0e9, -15.0);

// Close device
tgClose(device);
