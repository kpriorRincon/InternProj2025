import zmq
import pickle
import time

# Imports to run the sim
import Sig_Gen as Sig_Gen
import Sig_Gen_Noise
import Receiver as Receiver
import Repeater as Repeater
from sim_qpsk_noisy_demod import sample_read_output
from scipy.signal import hilbert
import numpy as np

context = zmq.Context()

# Control socket
ctrl = context.socket(zmq.REP)
ctrl.bind("tcp://127.0.0.1:6002") # sets up a REQ-REP connection with the controller/computer

# ---------------------------------------------------

# Getting the signal from the transmitter

print("Waiting for request from controller...")
req = ctrl.recv_string()
print("Request received.")
ctrl.send_string("Request received")
time.sleep(0.5)
with open('tx_to_rep.pkl', 'rb') as infile:
    tx = pickle.load(infile)
print("Signal acquired.")

# ---------------------------------------------------

# Modulating the signal and sending data back to the controller 
print("Waiting for request from controller...")
freq = ctrl.recv_json()
f_out = freq['freq out']
f_in = freq['freq in']
symbol_rate = 10e6
f_sample = 4e9 
incoming_qpsk = tx['tx signal']
t = tx['time']
repeater = Repeater.Repeater(sampling_frequency=f_sample)
repeater.desired_frequency = f_out

qpsk_mixed = np.real(repeater.mix(qpsk_signal=incoming_qpsk, qpsk_frequency=f_in, t=t))
symbol_rate *= f_out / f_in
f_sample *= f_out / f_in
qpsk_filtered = repeater.filter(f_out + 20e6, qpsk_mixed, order=10)
repeater.gain = 2
qpsk_amp = repeater.amplify(input_signal=qpsk_filtered)

rep = {"Incoming Signal": incoming_qpsk,
       "Mixed Signal": qpsk_mixed,
       "Filtered Signal": qpsk_filtered,
       "Outgoing Signal": qpsk_amp}

print("Sending data to controller...")
ctrl.send_pyobj(rep)
print("Data sent.")

# ---------------------------------------------------

# Send the QPSK signal to the receiver

print("Waiting for request from controller...")
req = ctrl.recv_string()
print("Request received.")
ctrl.send_string("Request received")

rep_to_rx = {"time": t,
             "rep signal": qpsk_amp}

with open('rep_to_rx.pkl','wb') as outfile:
    pickle.dump(rep_to_rx, outfile)

print("Signal sent.")

ctrl.close()
context.term()