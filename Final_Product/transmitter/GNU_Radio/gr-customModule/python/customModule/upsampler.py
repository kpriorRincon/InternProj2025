#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Rincon Research Interns 2025.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr

class upsampler(gr.interp_block):
    """
    Interpolator: Will take a complex signal at 1 sps and upsample to a desired sps. Zeros will be inserted between symbols to achieve the desired sps.
    Parameters:
        sps (int): Desired samples per symbol (interpolation factor).
    Input: complex64 stream
    Output: complex64 stream
    
    """
    def __init__(self, sps):
        self.sps = sps
        gr.interp_block.__init__(self,
            name="upsampler",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
            interp=self.sps)
       

    def work(self, input_items, output_items):
        symbols = input_items[0]
        upsampled_symbols = output_items[0]
        
        upsampled_symbols[:] = 0
        upsampled_symbols[::self.sps] = symbols

        return len(upsampled_symbols)
