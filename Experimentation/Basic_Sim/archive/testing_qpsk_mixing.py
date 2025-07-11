import Sig_Gen as Sig_Gen
import numpy as np
import matplotlib.pyplot as plt
sig_gen = Sig_Gen.SigGen()
sig_gen.freq = 910e6
sig_gen.sample_rate = 4e9
sig_gen.symbol_rate = 10e6

user_input = input("Enter a message to be sent: ")
message_bits = sig_gen.message_to_bits(user_input)
t,qpsk,_,_ = sig_gen.generate_qpsk(message_bits)

# the output qpsk is cos(2pifc+phi(t))
