#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: vsg experimentation
# GNU Radio version: 3.10.1.1

from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
import time
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import soapy
import vsg60




class data_to_vsg(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "vsg experimentation", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 2.88e6
        self.freq_tx = freq_tx = 910e6
        self.freq_rx = freq_rx = 920e6

        ##################################################
        # Blocks
        ##################################################
        self.vsg60_iqin_0 = vsg60.iqin(freq_tx, 0, samp_rate, False)
        self.soapy_bladerf_source_0 = None
        dev = 'driver=bladerf'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        self.soapy_bladerf_source_0 = soapy.source(dev, "fc32", 1, '',
                                  stream_args, tune_args, settings)
        self.soapy_bladerf_source_0.set_sample_rate(0, samp_rate)
        self.soapy_bladerf_source_0.set_bandwidth(0, 0.0)
        self.soapy_bladerf_source_0.set_frequency(0, freq_tx)
        self.soapy_bladerf_source_0.set_frequency_correction(0, 0)
        self.soapy_bladerf_source_0.set_gain(0, min(max(30.0, -1.0), 60.0))
        self.soapy_bladerf_sink_0 = None
        dev = 'driver=bladerf'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        self.soapy_bladerf_sink_0 = soapy.sink(dev, "fc32", 1, '',
                                  stream_args, tune_args, settings)
        self.soapy_bladerf_sink_0.set_sample_rate(0, samp_rate)
        self.soapy_bladerf_sink_0.set_bandwidth(0, 0.0)
        self.soapy_bladerf_sink_0.set_frequency(0, freq_rx)
        self.soapy_bladerf_sink_0.set_frequency_correction(0, 0)
        self.soapy_bladerf_sink_0.set_gain(0, min(max(40, 17.0), 73.0))
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/empire/Documents/InternProj2025/Final_Product/transmitter/data_for_sighound.bin', True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.vsg60_iqin_0, 0))
        self.connect((self.soapy_bladerf_source_0, 0), (self.soapy_bladerf_sink_0, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.soapy_bladerf_sink_0.set_sample_rate(0, self.samp_rate)
        self.soapy_bladerf_source_0.set_sample_rate(0, self.samp_rate)
        self.vsg60_iqin_0.set_srate(self.samp_rate)

    def get_freq_tx(self):
        return self.freq_tx

    def set_freq_tx(self, freq_tx):
        self.freq_tx = freq_tx
        self.soapy_bladerf_source_0.set_frequency(0, self.freq_tx)
        self.vsg60_iqin_0.set_frequency(self.freq_tx)

    def get_freq_rx(self):
        return self.freq_rx

    def set_freq_rx(self, freq_rx):
        self.freq_rx = freq_rx
        self.soapy_bladerf_sink_0.set_frequency(0, self.freq_rx)




def main(top_block_cls=data_to_vsg, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    #run for the fixed duration
    run_duration = 3 # in seconds
    time.sleep(run_duration)
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
