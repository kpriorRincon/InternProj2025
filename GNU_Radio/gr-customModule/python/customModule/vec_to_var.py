#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2025 Rincon Research Interns 2025.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr

class vec_to_var(gr.sync_block):
    """
    docstring for block vec_to_var
    """
    def __init__(self, vlen=1):
        gr.sync_block.__init__(self,
            name="vec_to_var",
            in_sig=[(np.complex64, vlen)],
            out_sig=None)
        self.vec = np.zeros(vlen, dtype=np.complex64)
        self.logger = gr.logger("vec_to_var")

    def work(self, input_items, output_items):
        in0 = input_items[0]
        
        self.logger.info("vec_to_var received this vector:", in0)

        self.logger.info("vec_to_var saved this vector:", self.vec)

        self.vec = np.array(in0[-1])

        self.logger.info("vec_to_var saved this vector:", self.vec)

        return len(input_items[0])
