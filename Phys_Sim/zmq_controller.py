# Scipts for a ZeroMQ system that implments the bent-pipe communication
# This script acts as the controller/computer that sends commands to the different nodes

import zmq
import time
import pickle

context = zmq.Context()

# Controller sockets
tx = context.socket(zmq.REQ)
tx.connect("tcp://127.0.0.1:6001") # sets up a REQ-REP connection with the transmitter
tx.setsockopt(zmq.LINGER, 0)

rep = context.socket(zmq.REQ)
rep.connect("tcp://127.0.0.1:6002") # sets up a REQ-REP connection with the repeater
rep.setsockopt(zmq.LINGER, 0)

rx = context.socket(zmq.REQ)
rx.connect("tcp://127.0.0.1:6003") # sets up a REQ-REP connection with the receiver
rx.setsockopt(zmq.LINGER, 0)

# ---------------------------------------------------

# Get user input
time.sleep(1)
print("Controller: Waiting for user input...")
# message = input("Enter the message you want to send:\n")
# f_in = float(input("Enter the desired frequency (in MHz) to transmit the message:\n"))
# f_in = f_in*1000000
# f_out = float(input("Enter the desired frequency (in MHz) to receive the message:\n"))
# f_out = f_out*1000000

with open('data_dict.pkl', 'rb') as infile:
    init_data = pickle.load(infile)

f_in = init_data['fin']
f_out = init_data['fout']
message = init_data['message']

# ---------------------------------------------------

# Send a request to the transmitter to get information and start encoding a message

# Sending request to transmitter
print("Controller: Sending request to transmitter...")
input_data = {"message": message, "freq in": f_in, "freq out": f_out}
tx.send_json(input_data)
print("Controller: Request sent.")

# Receiving response from transmitter
print("Controller: Waiting for response from transmitter...")
tx_data = tx.recv_pyobj()
print("Controller: Response received.")

# Data from transmitter
t = tx_data['time']
tx_qpsk = tx_data['qpsk']
tx_vert_lines = tx_data['vertical lines']
tx_symbols = tx_data['symbols']
message_bin = tx_data['message in binary']

# ---------------------------------------------------

# Send a request to the repeater to start listening, the transmitter to send the QPSK signal to the repeater, and the repeater to modulate the signal 

# Sending request to the repeater to start listening
print("Controller: Sending request to repeater...")
req = "Start listening."
rep.send_string(req)
print("Controller: Request sent.")
print("Controller: Waiting for response from repeater...")
rep.recv_string()
print("Controller: Response received")

# Sending request to transmitter to start sending the signal
print("Controller: Sending request to transmitter...")
req = "Transmit to repeater."
tx.send_string(req)
print("Controller: Request sent.")
print("Controller: Waiting for response from transmitter...")
tx.recv_string()
print("Controller: Response received")

# Sending request to the repeater to start modulating the signal
print("Controller: Sending request to repeater...")
rep_input = {"freq in": f_in, "freq out": f_out}
rep.send_json(rep_input)
print("Controller: Request sent.")

# Receiving data from the repeater
print("Controller: Waiting for response from repeater...")
rep_data = rep.recv_pyobj()
print("Controller: Response received.")

# Data from repeater
rep_incoming_signal = rep_data['Incoming Signal']
rep_mixed_signal = rep_data['Mixed Signal']
rep_filtered_signal = rep_data['Filtered Signal']
rep_outgoing_signal = rep_data['Outgoing Signal']

# ---------------------------------------------------

# Send a request to the receiver to start listening, the repeater to send the signal to the receiver, and the receiver to demodulate/decode the signal

# Sending request to the receiver to start listening
print("Controller: Sending request to receiver...")
req = "Start listening."
rx.send_string(req)
print("Controller: Request sent.")
print("Controller: Waiting for response from receiver...")
rx.recv_string()
print("Controller: Response received")

# Sending request to repeater to start sending the signal
print("Controller: Sending request to repeater...")
req = "Transmit to receiver."
rep.send_string(req)
print("Controller: Request sent.")
print("Controller: Waiting for response from repeater...")
rep.recv_string()
print("Controller: Response received")

# Sending request to the receiver to start demodulating/decoding the signal
print("Controller: Sending request to receiver...")
rx_input = {"freq out": f_out}
rx.send_json(rep_input)
print("Controller: Request sent.")

# Receiving data from the receiver
print("Controller: Waiting for response from receiver...")
rx_data = rx.recv_pyobj()
print("Controller: Response received.")

# Data from receiver
bit_sequence = rx_data['bit sequence']
recovered_message = rx_data['recovered message']
rx_incoming_signal = rx_data['incoming signal']
rx_filtered_signal = rx_data['filtered signal']
rx_analytical_signal = rx_data['analytical signal']

print('Controller: Original Message:', message)
print('Controller: Recovered Message:', recovered_message)

controller_data = {'time': t, 
                   'transmitter signal': tx_qpsk,
                   'transmitter vertical lines': tx_vert_lines,
                   'transmitter symbols': tx_symbols,
                   'transmitter message in binary': message_bin,
                   'repeater incoming signal': rep_incoming_signal,
                   'repeater mixed signal': rep_mixed_signal,
                   'repeater filtered signal': rep_filtered_signal,
                   'repeater outgoing signal': rep_outgoing_signal,
                   'receiver message in binary': bit_sequence,
                   'receiver recovered message': recovered_message,
                   'receiver incoming signal': rx_incoming_signal,
                   'receiver filtered signal': rx_filtered_signal,
                   'receiver analytical signal': rx_analytical_signal,
                   'freq in': f_in,
                   'freq out': f_out
                   }


with open('controller_data.pkl','wb') as outfile:
    pickle.dump(controller_data, outfile)

tx.close()
rep.close()
rx.close()
context.term()

