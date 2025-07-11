from gnuradio import gr, blocks
import pmt

class passthrough(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self)

        self.src = blocks.file_source(gr.sizeof_char, "bits_to_send.bin", False)
        self.sink = blocks.file_sink(gr.sizeof_char, "bits_read_in.bin", False)
        self.src.set_begin_tag(pmt.PMT_NIL)

        self.connect(self.src, self.sink)

if __name__ == '__main__':
    tb = passthrough()
    tb.start()
    tb.wait()  # Important: Wait ensures file_sink completes
