# Scipts for a ZeroMQ system that implments the bent-pipe communication
# This script acts as the receiver that receives commands from the controller

import zmq
import time
import pickle

# Imports to run the sim
import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater
from sim_qpsk_noisy_demod import sample_read_output
from scipy.signal import hilbert
import numpy as np

context = zmq.Context()

# Control socket
ctrl = context.socket(zmq.REP)
ctrl.bind("tcp://127.0.0.1:6003") # sets up a REQ-REP connection with the controller/computer
ctrl.setsockopt(zmq.LINGER, 0)

# ---------------------------------------------------

# Getting the signal from the repeater

print("Receiver: Waiting for request from controller...")
req = ctrl.recv_string()
print("Receiver: Request received.")
ctrl.send_string("Request received")
time.sleep(0.5)
with open('rep_to_rx.pkl', 'rb') as infile:
    rep = pickle.load(infile)
print("Receiver: Signal acquired.")

# ---------------------------------------------------

# Demodulating/decoding the signal
print("Receiver: Waiting for request from controller...")
freq = ctrl.recv_json()
f_out = freq["freq out"]
print("Receiver: Request received.")

symbol_rate = 10e6
f_sample = 4e9 

receiver = Receiver.Receiver(f_sample)

rep_signal = rep["rep signal"]
analytical_signal, bits = receiver.demodulator(rep_signal, f_sample, symbol_rate, f_out)
message = receiver.get_string(bits)

rep = {"bit sequence": bits,
       "recovered message": message}

print("Receiver: Sending data to controller...")
ctrl.send_pyobj(rep)
print("Receiver: Data sent.")

ctrl.close()
context.term()