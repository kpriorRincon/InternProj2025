#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
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

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import blocks
from gnuradio import customModule
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import soapy
import numpy as np
import vsg60



from gnuradio import qtgui

class hardwaretest(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
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

        self.settings = Qt.QSettings("GNU Radio", "hardwaretest")

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
        self.samp_rate = samp_rate = 2.88e6
        self.modulated_marker = modulated_marker = 0
        self.marker_offset = marker_offset = tuple([0]*(int(skip/2)+1))
        self.marker = marker = (1,1,1,1,1,0,0,1,1,0,1,0,0,1,0,0,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1)
        self.freq = freq = 905e6
        self.data = data = (0,1,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1)
        self.alpha = alpha = 0.35

        ##################################################
        # Blocks
        ##################################################
        self.vsg60_iqin_0 = vsg60.iqin(freq, -10, samp_rate, False)
        self.soapy_rtlsdr_source_0 = None
        dev = 'driver=rtlsdr'
        stream_args = ''
        tune_args = ['']
        settings = ['']

        self.soapy_rtlsdr_source_0 = soapy.source(dev, "fc32", 1, '',
                                  stream_args, tune_args, settings)
        self.soapy_rtlsdr_source_0.set_sample_rate(0, samp_rate)
        self.soapy_rtlsdr_source_0.set_gain_mode(0, False)
        self.soapy_rtlsdr_source_0.set_frequency(0, freq)
        self.soapy_rtlsdr_source_0.set_frequency_correction(0, 0)
        self.soapy_rtlsdr_source_0.set_gain(0, 'TUNER', 20)
        self.root_raised_cosine_filter_1 = filter.interp_fir_filter_ccf(
            1,
            firdes.root_raised_cosine(
                1,
                samp_rate,
                samp_rate/sps,
                alpha,
                num_taps))
        self.root_raised_cosine_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.root_raised_cosine(
                1,
                samp_rate,
                samp_rate/sps,
                alpha,
                num_taps))
        self.qtgui_time_sink_x_0_0 = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "Rx", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0_0.enable_tags(True)
        self.qtgui_time_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0_0.enable_grid(False)
        self.qtgui_time_sink_x_0_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0_0.enable_stem_plot(False)


        labels = ['Real', 'Imaginary', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_0_win)
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "Tx", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['Real', 'Imaginary', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.qtgui_const_sink_x_0_0 = qtgui.const_sink_c(
            1024, #size
            "Tx", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0_0.enable_grid(False)
        self.qtgui_const_sink_x_0_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_0_win)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            "Rx", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(False)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_win)
        self.customModule_upsampler_0 = customModule.upsampler(sps)
        self.customModule_downsampler_0 = customModule.downsampler(sps)
        self.customModule_QPSK_Modulator_0 = customModule.QPSK_Modulator()
        self.customModule_QPSK_Demodulator_0 = customModule.QPSK_Demodulator()
        self.blocks_vector_source_x_0 = blocks.vector_source_i(marker+data, True, 1, [])
        self.blocks_skiphead_0_0_0_0 = blocks.skiphead(gr.sizeof_int*1, group_delay)
        self.blocks_skiphead_0_0 = blocks.skiphead(gr.sizeof_gr_complex*1, skip)
        self.blocks_skiphead_0 = blocks.skiphead(gr.sizeof_gr_complex*1, skip)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_int*1)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_skiphead_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.vsg60_iqin_0, 0))
        self.connect((self.blocks_skiphead_0_0, 0), (self.customModule_downsampler_0, 0))
        self.connect((self.blocks_skiphead_0_0_0_0, 0), (self.blocks_null_sink_0, 0))
        self.connect((self.blocks_vector_source_x_0, 0), (self.customModule_QPSK_Modulator_0, 0))
        self.connect((self.customModule_QPSK_Demodulator_0, 0), (self.blocks_skiphead_0_0_0_0, 0))
        self.connect((self.customModule_QPSK_Modulator_0, 0), (self.customModule_upsampler_0, 0))
        self.connect((self.customModule_QPSK_Modulator_0, 0), (self.qtgui_const_sink_x_0_0, 0))
        self.connect((self.customModule_downsampler_0, 0), (self.customModule_QPSK_Demodulator_0, 0))
        self.connect((self.customModule_downsampler_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.customModule_upsampler_0, 0), (self.root_raised_cosine_filter_1, 0))
        self.connect((self.root_raised_cosine_filter_0, 0), (self.blocks_skiphead_0_0, 0))
        self.connect((self.root_raised_cosine_filter_1, 0), (self.blocks_skiphead_0, 0))
        self.connect((self.soapy_rtlsdr_source_0, 0), (self.qtgui_time_sink_x_0_0, 0))
        self.connect((self.soapy_rtlsdr_source_0, 0), (self.root_raised_cosine_filter_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "hardwaretest")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_num_taps(self):
        return self.num_taps

    def set_num_taps(self, num_taps):
        self.num_taps = num_taps
        self.set_group_delay(int(self.num_taps/2))
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))
        self.root_raised_cosine_filter_1.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_skip(int((self.group_delay/2)*(self.sps/2)))
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))
        self.root_raised_cosine_filter_1.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))

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
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.qtgui_time_sink_x_0_0.set_samp_rate(self.samp_rate)
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))
        self.root_raised_cosine_filter_1.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))
        self.soapy_rtlsdr_source_0.set_sample_rate(0, self.samp_rate)
        self.vsg60_iqin_0.set_srate(self.samp_rate)

    def get_modulated_marker(self):
        return self.modulated_marker

    def set_modulated_marker(self, modulated_marker):
        self.modulated_marker = modulated_marker

    def get_marker_offset(self):
        return self.marker_offset

    def set_marker_offset(self, marker_offset):
        self.marker_offset = marker_offset

    def get_marker(self):
        return self.marker

    def set_marker(self, marker):
        self.marker = marker
        self.blocks_vector_source_x_0.set_data(self.marker+self.data, [])

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.soapy_rtlsdr_source_0.set_frequency(0, self.freq)
        self.vsg60_iqin_0.set_frequency(self.freq)

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data
        self.blocks_vector_source_x_0.set_data(self.marker+self.data, [])

    def get_alpha(self):
        return self.alpha

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.root_raised_cosine_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))
        self.root_raised_cosine_filter_1.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate/self.sps, self.alpha, self.num_taps))




def main(top_block_cls=hardwaretest, options=None):

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
