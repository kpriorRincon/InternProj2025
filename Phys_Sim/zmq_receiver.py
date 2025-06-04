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

# ---------------------------------------------------

# Getting the signal from the repeater

print("Waiting for request from controller...")
req = ctrl.recv_string()
print("Request received.")
ctrl.send_string("Request received")
time.sleep(0.5)
with open('rep_to_rx.pkl', 'rb') as infile:
    rep = pickle.load(infile)
print("Signal acquired.")

# ---------------------------------------------------

# Demodulating/decoding the signal

print("Waiting for request from controller...")
freq = ctrl.recv_json()
f_out = freq["freq out"]
print("Request received.")

symbol_rate = 10e6
f_sample = 4e9 

_, bits = Receiver.demodulator(rep["rep signal"],f_sample, symbol_rate,f_out)
message = Receiver.get_string(bits)

rep = {"bit sequence": bits,
       "recovered message": message}

print("Sending data to controller...")
ctrl.send_pyobj(rep)
print("Data sent.")