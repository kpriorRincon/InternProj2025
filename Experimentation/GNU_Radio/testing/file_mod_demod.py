#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: file_mod_demod
# Author: trevorwiseman
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
import pmt
from gnuradio import digital
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



from gnuradio import qtgui

class file_mod_demod(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "file_mod_demod", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("file_mod_demod")
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

        self.settings = Qt.QSettings("GNU Radio", "file_mod_demod")

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
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################
        self.interp_fir_filter_xxx_0 = filter.interp_fir_filter_ccc(1, [51])
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
        self.fir_filter_xxx_0 = filter.fir_filter_ccc(1, [51])
        self.fir_filter_xxx_0.declare_sample_delay(0)
        self.filter_fft_rrc_filter_0_0 = filter.fft_filter_ccc(1, firdes.root_raised_cosine(1, samp_rate, 1.0, 0.35, 11*samp_rate), 1)
        self.filter_fft_rrc_filter_0 = filter.fft_filter_ccc(1, firdes.root_raised_cosine(1, samp_rate, 1.0, 0.35, 11*samp_rate), 1)
        self.digital_psk_mod_0 = digital.psk.psk_mod(
            constellation_points=8,
            mod_code="gray",
            differential=True,
            samples_per_symbol=2,
            excess_bw=0.35,
            verbose=False,
            log=False)
        self.digital_psk_demod_0 = digital.psk.psk_demod(
            constellation_points=8,
            differential=True,
            samples_per_symbol=2,
            excess_bw=0.35,
            phase_bw=6.28/100.0,
            timing_bw=6.28/100.0,
            mod_code="gray",
            verbose=False,
            log=False)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, '/home/trevorwiseman/Documents/GitHub/InternProj2025/GNU_Radio/testing/bits_to_send.txt', True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_char*1, '/home/trevorwiseman/Documents/GitHub/InternProj2025/GNU_Radio/testing/bits_read_in.txt', False)
        self.blocks_file_sink_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.digital_psk_mod_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.interp_fir_filter_xxx_0, 0))
        self.connect((self.digital_psk_demod_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.digital_psk_mod_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.filter_fft_rrc_filter_0, 0), (self.filter_fft_rrc_filter_0_0, 0))
        self.connect((self.filter_fft_rrc_filter_0_0, 0), (self.fir_filter_xxx_0, 0))
        self.connect((self.fir_filter_xxx_0, 0), (self.digital_psk_demod_0, 0))
        self.connect((self.interp_fir_filter_xxx_0, 0), (self.filter_fft_rrc_filter_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "file_mod_demod")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.filter_fft_rrc_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, 1.0, 0.35, 11*self.samp_rate))
        self.filter_fft_rrc_filter_0_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, 1.0, 0.35, 11*self.samp_rate))




def main(top_block_cls=file_mod_demod, options=None):

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
