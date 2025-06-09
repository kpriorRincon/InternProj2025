# Scipts for a ZeroMQ system that implments the bent-pipe communication
# This script acts as the receiver that receives commands from the controller

import zmq
import time
import pickle

# Imports to run the sim
import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater
from scipy.signal import hilbert
import numpy as np

def Noise_Addr(sinusoid, noise_power):
        import numpy as np
        # Noise parameters
        mean_noise = 0          # Mean of the noise distribution
        std_noise = noise_power # Standard deviation of the noise distribution

        # Generate noise
        noise_real = np.random.normal(mean_noise, std_noise/np.sqrt(2), len(sinusoid))
        noise_imag = np.random.normal(mean_noise, std_noise/np.sqrt(2), len(sinusoid))
        noise = noise_real + 1j*noise_imag
        return sinusoid + noise                    

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

# symbol_rate = 10e6
# f_sample = 4e9 

with open('data_dict.pkl', 'rb') as infile:
    init_data = pickle.load(infile)

f_sample = init_data['sample rate']
symbol_rate = init_data['symbol rate']
noise_bool = init_data['noise_bool']
noise_power = init_data['noise_power']

receiver = Receiver.Receiver(f_sample)
if noise_bool:
    rep_signal = Noise_Addr(rep["rep signal"], noise_power)
else:
    rep_signal = rep["rep signal"]

signal, sampled_symbols, bits = receiver.demodulator(rep_signal, f_sample, symbol_rate, f_out)
message = receiver.get_string(bits)

rep = {"bit sequence": bits,
       "recovered message": message,
       "incoming signal": rep_signal,
       "filtered signal": rep_signal, 
       'analytical signal': signal,
       'sampled symbols': sampled_symbols
       }

print("Receiver: Sending data to controller...")
ctrl.send_pyobj(rep)
print("Receiver: Data sent.")

ctrl.close()
context.term()