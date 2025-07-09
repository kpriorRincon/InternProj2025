#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Rincon Research Interns 2025.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr

class downsampler(gr.decim_block):
    """
    Decimator: Will take a complex signal at a sps and downsample to a 1 sps.
    Parameters:
        sps (int): Current samples per symbol (decimation factor).
    Input: complex64 stream
    Output: complex64 stream

    """
    def __init__(self, sps):
        self.sps = sps
        gr.decim_block.__init__(self,
            name="downsampler",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
            decim=self.sps)


    def work(self, input_items, output_items):
        upsampled_symbols = input_items[0]
        symbols = output_items[0]
        
        symbols[:] = upsampled_symbols[::self.sps]

        return len(symbols)
