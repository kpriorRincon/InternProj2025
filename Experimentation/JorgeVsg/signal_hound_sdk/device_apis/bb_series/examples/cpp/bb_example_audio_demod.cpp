/** [audioDemodExample1] */

#include "bb_api.h"
#include <iostream>
#include <vector>

#ifdef _WIN32
#pragma comment(lib, "bb_api.lib")
#endif

/*
This example demonstrates how to configure, initialize, and retrieve data
in audio demodulation mode.
*/

void bbExampleAudioDemod()
{
    // Connect device
    int handle;
    bbStatus status = bbOpenDevice(&handle);
    if(status != bbNoError) {
        std::cout << "Issue opening device\n";
        std::cout << bbGetErrorString(status) << "\n";
        return;
    }

    // Configure the measurement parameters

	// Set the demodulation scheme, center frequency, IF bandwidth,
	//   post demodulation low & hi pass filter frequencies, and FM deemphasis in microseconds
	status = bbConfigureDemod(handle, BB_DEMOD_FM, 97.1e6, 120.0e3, 8.0e3, 20.0, 75.0);

    // Initiate the device, once this function returns the device
    //   will be streaming demodulated audio data.
    status = bbInitiate(handle, BB_AUDIO_DEMOD, 0);
    if(status != bbNoError) {
        std::cout << "Initiate error\n";
        std::cout << bbGetErrorString(status) << "\n";
        bbCloseDevice(handle);
        return;
    }

    // Allocate memory for 4096 audio samples for an audio sample rate of 32k.
    // This is the number of audio samples we will acquire each function call.
    std::vector<float> buffer(4096);

    // Perform the capture
    status = bbFetchAudio(handle, buffer.data());
    // Check status here

    // At this point, 4096 audio samples have been retrieved and
    //   stored in the buffer array. Any processing on the data should happen here.

    // Continue to call bbFetchAudio.
	// While streaming, it is possible to continue to change the audio settings via 
	//  bbConfigureDemod() as long as the updated center frequency is not +/- 8 MHz 
	//  of the value specified when bbInitiate() was called.

    for(int i = 0; i < 10; i++) {
        status = bbFetchAudio(handle, buffer.data());
        // Check status here

        // Do processing
    }

    // When done, stop streaming and close device.
    bbAbort(handle);
    bbCloseDevice(handle);
}

/** [audioDemodExample1] */
