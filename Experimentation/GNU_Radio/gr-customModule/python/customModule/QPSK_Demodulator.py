#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Rincon Research Interns 2025.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr

class QPSK_Demodulator(gr.interp_block):
    """
    QPSK Demodulator: Maps complex symbols to bits (2 bits per symbol). The ouput will have double the amount of items as the input.
    Input: complex64 stream
    Output: uint32 bitstream
    """
    def __init__(self):
        gr.interp_block.__init__(self,
            name="QPSK_Demodulator",
            in_sig=[np.complex64],
            out_sig=[np.uint32], 
            interp = 2)


    def work(self, input_items, output_items):
        symbols = input_items[0]
        bits = output_items[0]

        num_symbols = len(symbols)
        num_bits = len(symbols) * 2

        for i in range(num_symbols):
            complex_number = symbols[i]
            real = np.real(complex_number)
            imag = np.imag(complex_number)

            # Determine bits based on quadrant
            if real > 0 and imag > 0:
                bit1 = 0
                bit2 = 0
            elif real < 0 and imag > 0:
                bit1 = 0
                bit2 = 1
            elif real > 0 and imag < 0:
                bit1 = 1
                bit2 = 0
            elif real < 0 and imag < 0:
                bit1 = 1
                bit2 = 1

            bits[2 * i] = bit1
            bits[2 * i + 1] = bit2

        return len(bits)
