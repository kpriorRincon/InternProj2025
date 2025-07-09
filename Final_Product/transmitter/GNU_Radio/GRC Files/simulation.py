#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: simulation
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
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import numpy as np



from gnuradio import qtgui

class simulation(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "simulation", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("simulation")
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

        self.settings = Qt.QSettings("GNU Radio", "simulation")

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
        self.alpha = alpha = 0.35
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(1.0,samp_rate,samp_rate/sps,alpha,num_taps)
        self.offset = offset = tuple([0]*(int(skip/2)+1))
        self.modulated_marker = modulated_marker = (-0.162794-0.195436j, -0.205587-0.206878j, -0.244216-0.186516j, -0.254743-0.157722j, -0.209717-0.148513j, -0.0954317-0.176885j, 0.05619-0.22366j, 0.181965-0.247448j, 0.220588-0.208159j, 0.145461-0.102616j, -0.00375595+0.038455j, -0.14837+0.157788j, -0.213419+0.205581j, -0.175588+0.159695j, -0.0590324+0.0458357j, 0.0807639-0.0872715j, 0.195449-0.195449j, 0.257974-0.251467j, 0.272338-0.259142j, 0.247074-0.23118j, 0.185101-0.177262j, 0.0834665-0.0928842j, -0.0445037+0.00980465j, -0.156794+0.113949j, -0.205825+0.190738j, -0.158646+0.224972j, -0.038819+0.210409j, 0.101901+0.174055j, 0.207276+0.150231j, 0.242049+0.165696j, 0.224443+0.202897j, 0.188335+0.227514j, 0.162175+0.202416j, 0.159211+0.119185j, 0.161785+0.000727935j, 0.162007-0.116039j, 0.166769-0.194844j, 0.182772-0.224201j, 0.216931-0.215366j, 0.241793-0.188264j, 0.21436-0.163386j, 0.1127-0.156278j, -0.0403838-0.168198j, -0.181545-0.184669j, -0.23315-0.189987j, -0.159771-0.184669j, 0.00495681-0.168198j, 0.173208-0.156278j,
        0.247753-0.160727j, 0.17506-0.189106j, -0.000727928-0.219486j, -0.178206-0.227841j, -0.250008-0.191462j, -0.178206-0.112162j, -0.000727926-5.87783e-09j, 0.17506+0.112162j, 0.247753+0.19412j, 0.173208+0.226998j, 0.00495681+0.215366j, -0.159771+0.185467j, -0.230491+0.16411j, -0.182388+0.160156j, -0.0445037+0.16747j, 0.10906+0.177645j)
        self.marker_offset = marker_offset = tuple([0]*(int(skip/2)+1))
        self.marker = marker = (1,1,1,1,1,0,0,1,1,0,1,0,0,1,0,0,0,0,1,0,1,0,1,1,1,0,1,1,0,0,0,1)
        self.freq = freq = 905e6
        self.data = data = (0,1,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1)

        ##################################################
        # Blocks
        ##################################################
        self.customModule_QPSK_Modulator_0 = customModule.QPSK_Modulator()
        self.blocks_vector_source_x_0 = blocks.vector_source_i(marker+data, False, 1, [])
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_int*1, samp_rate,True)
        self.blocks_file_sink_0_0 = blocks.file_sink(gr.sizeof_gr_complex*1, '/home/empire/Documents/InternProj2025/GNU_Radio/testing/bits_read_in.bin', False)
        self.blocks_file_sink_0_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_throttle_0, 0), (self.customModule_QPSK_Modulator_0, 0))
        self.connect((self.blocks_vector_source_x_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.customModule_QPSK_Modulator_0, 0), (self.blocks_file_sink_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "simulation")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_num_taps(self):
        return self.num_taps

    def set_num_taps(self, num_taps):
        self.num_taps = num_taps
        self.set_group_delay(int(self.num_taps/2))
        self.set_rrc_taps(firdes.root_raised_cosine(1.0,self.samp_rate,self.samp_rate/self.sps,self.alpha,self.num_taps))

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_rrc_taps(firdes.root_raised_cosine(1.0,self.samp_rate,self.samp_rate/self.sps,self.alpha,self.num_taps))
        self.set_skip(int((self.group_delay/2)*(self.sps/2)))

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
        self.set_offset(tuple([0]*(int(self.skip/2)+1)))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_rrc_taps(firdes.root_raised_cosine(1.0,self.samp_rate,self.samp_rate/self.sps,self.alpha,self.num_taps))
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)

    def get_alpha(self):
        return self.alpha

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.set_rrc_taps(firdes.root_raised_cosine(1.0,self.samp_rate,self.samp_rate/self.sps,self.alpha,self.num_taps))

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps

    def get_offset(self):
        return self.offset

    def set_offset(self, offset):
        self.offset = offset

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

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data
        self.blocks_vector_source_x_0.set_data(self.marker+self.data, [])




def main(top_block_cls=simulation, options=None):
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")

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
