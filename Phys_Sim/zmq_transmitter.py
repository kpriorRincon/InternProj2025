import zmq
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
ctrl.bind("tcp://127.0.0.1:6001") # sets up a REQ-REP connection with the controller/computer

# Data socket
data = context.socket(zmq.PUSH)
data.bind("tcp://127.0.0.1:3001") # sets up a PUSH-PULL connection with the repeater

# ---------------------------------------------------

# Send a request to the transmitter to get information and start encoding a message

# Receiving request from controller
print("Waiting for request from controller...")
req = ctrl.recv_json()
print("Request received.")

# Extracting the message and frequencies from the request
message = req['message']
f_in = req['freq in']
f_out = req['freq out']

symbol_rate = 10e6
f_sample = 4e9 
sig_gen = Sig_Gen.SigGen(f_sample, symbol_rate)
sig_gen.freq = f_in
sig_gen.sample_rate = f_sample
sig_gen.symbol_rate = symbol_rate 

message_bin = sig_gen.message_to_bits(message)
print("Message converted to binary.")
sig_gen.generate_qpsk(message_bin)
print("QPSK signal generated.")

t = sig_gen.time_vector
tx_qpsk = sig_gen.qpsk_waveform
tx_vert_lines = sig_gen.time_vertical_lines
tx_symbols = sig_gen.symbols

rep = {"time": t, 
       "qpsk": tx_qpsk, 
       "vertical lines": tx_vert_lines, 
       "symbols": tx_symbols,
       "message in binary": message_bin}

# Sending the generated QPSK signal and data to the controller
print("Sending data to controller...")
ctrl.send_pyobj(rep)

# ---------------------------------------------------

# Send the QPSK signal to the repeater

print("Waiting for request from controller...")
req = ctrl.recv_string()
print("Request received.")
ctrl.send_string("Request received")

tx_to_rep = {"time": t,
             "tx signal": tx_qpsk}

with open('tx_to_rep.pkl','wb') as outfile:
    pickle.dump(tx_to_rep, outfile)

print("Signal sent.")

ctrl.close()
context.term()