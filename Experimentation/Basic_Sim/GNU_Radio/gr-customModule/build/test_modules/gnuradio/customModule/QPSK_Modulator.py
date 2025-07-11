#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Rincon Research Interns 2025.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr

class QPSK_Modulator(gr.decim_block):
    """
    QPSK Modulator: Maps bits to complex symbols (2 bits per symbol). The output will have half the amount of items as the input.
    Input: uint32 bitstream
    Output: complex64 stream
    """
    def __init__(self):
        gr.decim_block.__init__(self,
            name="QPSK_Modulator",
            in_sig=[np.uint32],
            out_sig=[np.complex64],
            decim = 2)


    def work(self, input_items, output_items):
        bits = input_items[0]
        symbols = output_items[0]
    
        num_bits = len(bits)
        num_symbols = num_bits // 2
        bits_to_process = num_symbols * 2

        # index of mapping represents what bit it maps to
        mapping =(1/np.sqrt(2)) * np.array([
             1 + 1j,  # 00
            -1 + 1j,  # 01
             1 - 1j,  # 10
            -1 - 1j   # 11
        ])
        
        for i in range(num_symbols):
            bit1 = bits[2 * i]
            bit2 = bits[2 * i + 1]
            symbol_index = bit1 * 2**1 + bit2 * 2**0
            symbols[i] = mapping[symbol_index]

        return len(symbols)
