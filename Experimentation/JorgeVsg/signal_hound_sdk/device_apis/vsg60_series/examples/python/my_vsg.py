import numpy as np

# -*- coding: utf-8 -*-

# This example generates a basic CW signal.

from vsgdevice.vsg_api import *
from time import sleep

def generate_iq():
    # Open device
    handle = vsg_open_device()["handle"]

    # Configure generator
    freq = 1.0e9 # Hz
    sample_rate = 50.0e6 # samples per second
    level = -20.0 # dBm

    vsg_set_frequency(handle, freq)
    vsg_set_level(handle, level)
    vsg_set_sample_rate(handle, sample_rate)

    # Output CW, single I/Q value of {1,0}
    # This is equivalent to calling vsgOutputCW
    num_samples = 128       # Number of samples in the waveform
    freq = 0.1              # Normalized frequency (cycles/sample)
    amplitude = 1.0         # Sine wave amplitude

    # Generate complex sine wave: I = cos(2πft), Q = sin(2πft)
    t = np.arange(num_samples)
    iq_complex = amplitude * np.exp(1j * 2 * np.pi * freq * t)

    # Interleave I and Q for vsg_repeat_waveform (assuming it expects [I0, Q0, I1, Q1, ...])
    iq = np.empty(num_samples * 2, dtype=np.float32)
    iq[0::2] = np.real(iq_complex)
    iq[1::2] = np.imag(iq_complex)


    vsg_repeat_waveform(handle, iq, 1)

    #vsg_output_waveform(device, iq, length):


    # Will transmit until you close the device or abort
    sleep(5)

    # Stop waveform
    vsg_abort(handle)

    # Done with device
    vsg_close_device(handle)

if __name__ == "__main__":
    generate_iq()
