# Scipts for a ZeroMQ system that implments the bent-pipe communication
# This script acts as the repeater that receives commands from the controller

import zmq
import pickle
import time

# Imports to run the sim
import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater
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
ctrl.bind("tcp://127.0.0.1:6002") # sets up a REQ-REP connection with the controller/computer
ctrl.setsockopt(zmq.LINGER, 0)

# ---------------------------------------------------

# Getting the signal from the transmitter

print("Repeater: Waiting for request from controller...")
req = ctrl.recv_string()
print("Repeater: Request received.")
ctrl.send_string("Request received")
time.sleep(0.5)
with open('tx_to_rep.pkl', 'rb') as infile:
    tx = pickle.load(infile)
print("Repeater: Signal acquired.")

# ---------------------------------------------------

# Modulating the signal and sending data back to the controller 
print("Repeater: Waiting for request from controller...")
freq = ctrl.recv_json()
f_out = freq['freq out']
f_in = freq['freq in']

# symbol_rate = 10e6
# f_sample = 4e9 

with open('data_dict.pkl', 'rb') as infile:
    init_data = pickle.load(infile)

f_sample = init_data['sample rate']
symbol_rate = init_data['symbol rate']
noise_bool = init_data['noise_bool']
noise_power = init_data['noise_power']
gain = init_data['gain']

if noise_bool:
    incoming_qpsk = Noise_Addr(tx['tx signal'], noise_power)
else:
    incoming_qpsk = tx['tx signal']

t = tx['time']
repeater = Repeater.Repeater(sampling_frequency=f_sample, symbol_rate=symbol_rate)
repeater.desired_frequency = f_out

qpsk_mixed = repeater.mix(qpsk_signal=incoming_qpsk, qpsk_frequency=f_in, t=t)
symbol_rate *= f_out / f_in
f_sample *= f_out / f_in
repeater.gain = gain
qpsk_amp = repeater.amplify(input_signal=qpsk_mixed)

rep = {"Incoming Signal": incoming_qpsk,
       "Mixed Signal": qpsk_mixed,
       "Outgoing Signal": qpsk_amp}

print("Repeater: Sending data to controller...")
ctrl.send_pyobj(rep)
print("Repeater: Data sent.")

# ---------------------------------------------------

# Send the QPSK signal to the receiver

print("Repeater: Waiting for request from controller...")
req = ctrl.recv_string()
print("Repeater: Request received.")
ctrl.send_string("Request received")

rep_to_rx = {"time": t,
             "rep signal": qpsk_mixed}

with open('rep_to_rx.pkl','wb') as outfile:
    pickle.dump(rep_to_rx, outfile)

print("Repeater: Signal sent.")

ctrl.close()
context.term()