#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Trevor Wiseman.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr
from scipy.signal import fftconvolve

class correlator(gr.sync_block):
    """
    docstring for block correlator
    """
    def __init__(self, sps=2):
        gr.sync_block.__init__(self,
            name="correlator",
            in_sig=[np.complex64, np.complex64, np.complex64],
            out_sig=[np.complex64])
        self.sps = sps  # samples per symbol  

    def work(self, input_items, output_items):
        # get inputs
        rx_signal = input_items[0][:]
        start_marker = input_items[1][:]
        end_marker = input_items[2][:]

        # initial indices
        start_index = 0
        end_index = len(rx_signal) - 1

        if rx_signal.size > 0:
            # correlate
            correlated_signal = fftconvolve(rx_signal, np.conj(np.flip(start_marker)), mode='full')
            end_cor_signal = fftconvolve(rx_signal, np.conj(np.flip(end_marker)), mode='full')

            # get indices
            start_index = np.argmax(np.abs(correlated_signal)) - 16*self.sps # go back 16 symbols e.g. 32 bits
            
            if start_index > end_index:
                start_index = 0
            else:
                end_index = np.argmax(np.abs(end_cor_signal))

        # return the signal at those indices
        output_items[0] = rx_signal[start_index:end_index]
        return len(output_items[0])
