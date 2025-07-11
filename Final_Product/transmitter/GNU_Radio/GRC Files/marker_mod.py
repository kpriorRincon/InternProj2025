#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: marker_mod
# GNU Radio version: 3.10.1.1

from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from gnuradio import blocks
from gnuradio import customModule
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import numpy as np



from gnuradio import qtgui

class marker_mod(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "marker_mod", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("marker_mod")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "marker_mod")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.num_taps = num_taps = int(40)
        self.sps = sps = 4
        self.group_delay = group_delay = int(num_taps/2)
        self.skip = skip = int((group_delay/2)*(sps/2))
        self.samp_rate = samp_rate = 32000
        self.marker_offset = marker_offset = tuple([0]*(int(skip/2)+1))
        self.marker = marker = (1,1,1,1,1,0,0,1,1,0,1,0,0,1,0,0,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1)
        self.data = data = (0,1,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1)
        self.alpha = alpha = 0.35

        ##################################################
        # Blocks
        ##################################################
        self.root_raised_cosine_filter_1_0 = filter.interp_fir_filter_ccf(
            1,
            firdes.root_raised_cosine(
                1,
                samp_rate,
                samp_rate/sps,
                alpha,
                num_taps))
        self.customModule_upsampler_0_0 = customModule.upsampler(sps)
        self.customModule_QPSK_Modulator_0_0 = customModule.QPSK_Modulator()
        self.blocks_vector_source_x_1 = blocks.vector_source_i(marker+marker_offset, False, 1, [])
        self.blocks_throttle_0_0 = blocks.throttle(gr.sizeof_int*1, samp_rate,True)
        self.blocks_skiphead_0_0 = blocks.skiphead(gr.sizeof_gr_complex*1, skip)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, '/home/empire/Documents/InternProj2025/GNU_Radio/testing/bits_read_in.bin', False)
        self.blocks_file_sink_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_skiphead_0_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_throttle_0_0, 0), (self.customModule_QPSK_Modulator_0_0, 0))
        self.connect((self.blocks_vector_source_x_1, 0), (self.blocks_throttle_0_0, 0))
        self.connect((self.customModule_QPSK_Modulator_0_0, 0), (self.customModule_upsampler_0_0, 0))
        self.connect((self.customModule_upsampler_0_0, 0), (self.root_raised_cosine_filter_1_0, 0))
        self.connect((self.root_raised_cosine_filter_1_0, 0), (self.blocks_skiphead_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "marker_mod")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_num_taps(self):
        return self.num_taps

    def set_num_taps(self, num_taps):
        self.num_taps = num_taps
        self.set_group_delay(int(self.num_taps/2))
        self.root_raised_cosine_filter_1_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_skip(int((self.group_delay/2)*(self.sps/2)))
        self.root_raised_cosine_filter_1_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))

    def get_group_delay(self):
        return self.group_delay

    def set_group_delay(self, group_delay):
        self.group_delay = group_delay
        self.set_skip(int((self.group_delay/2)*(self.sps/2)))

    def get_skip(self):
        return self.skip

    def set_skip(self, skip):
        self.skip = skip
        self.set_marker_offset(tuple([0]*(int(self.skip/2)+1)))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0_0.set_sample_rate(self.samp_rate)
        self.root_raised_cosine_filter_1_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))

    def get_marker_offset(self):
        return self.marker_offset

    def set_marker_offset(self, marker_offset):
        self.marker_offset = marker_offset
        self.blocks_vector_source_x_1.set_data(self.marker+self.marker_offset, [])

    def get_marker(self):
        return self.marker

    def set_marker(self, marker):
        self.marker = marker
        self.blocks_vector_source_x_1.set_data(self.marker+self.marker_offset, [])

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def get_alpha(self):
        return self.alpha

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.root_raised_cosine_filter_1_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))




def main(top_block_cls=marker_mod, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
