"""
===============================================================================
File Name     : Tx.py
Description   : collect audio samples modulate them using 16-bit QAM and transmit them
Authors       : Skylar Harris, Jorge Hernandez, Kobe Prior, Trevor Wiseman=
Created       : 2025-07-15
Last Modified : 2025-07-18
Version       : 1.0
Python Version: 3.x
===============================================================================

Notes:
- Any additional context or usage instructions.
- External dependencies (e.g., numpy, matplotlib).
- Reference links or documentation sources if applicable.

Example Usage:
    $ python Tx.py

===============================================================================
"""

# Library Imports
import wave
import pyaudio
import numpy as np
import SigGen as SigGen
from vsgdevice.vsg_api import *
import time
#defining Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 # if sys.platform == 'darwin' else 2
RATE = 44100
RECORD_SECONDS = 5
sig_gen = SigGen.SigGen(910e6, 1)
#initialize the VSG device
print('Opening VSG Device...')
# Open device
handle = vsg_open_device()["handle"]

#Record the Audio
with wave.open('transmit.wav', 'wb') as wf:
    p = pyaudio.PyAudio()
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

    print('Recording...')
    for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
        wf.writeframes(stream.read(CHUNK))
    print('Done')

    stream.close()
    p.terminate()


#Convert the Audio to IQ data
IQ_data = np.array([], dtype=np.complex64)
#iterates through everything in transmit.wav and puts it into iq ready for transmission
with wave.open('transmit.wav', 'rb') as wave_file:
    print('Converting to IQ...')
    while True:
        d = wave_file.readframes(CHUNK)
        if not d:
            break

        # Convert to bits
        d_bits = np.unpackbits(np.frombuffer(d, dtype=np.uint8))

        # Modulate
        _, qam_sig = sig_gen.generate_16QAM(d_bits)
        IQ_data.append(qam_sig)
print('Signal Ready to Transmit')

#Send IQ data with the VSG

iq = np.empty(IQ_data.size * 2, dtype=np.float32)
iq[0::2] = IQ_data.real
iq[1::2] = IQ_data.imag
vsg_repeat_waveform(handle, iq, len(iq));

# Will transmit until you close the device or abort
time.sleep(25)
# Stop waveform
vsg_abort(handle);

# Done with device
vsg_close_device(handle);