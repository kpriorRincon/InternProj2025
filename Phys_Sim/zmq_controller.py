# Scipts for a ZeroMQ system that implments the bent-pipe communication
# This script acts as the controller/computer that sends commands to the different nodes

import zmq
import time

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
message = input("Enter the message you want to send:\n")
f_in = float(input("Enter the desired frequency (in MHz) to transmit the message:\n"))
f_in = f_in*1000000
f_out = float(input("Enter the desired frequency (in MHz) to receive the message:\n"))
f_out = f_out*1000000

# ---------------------------------------------------

# Send a request to the transmitter to get information and start encoding a message

# Sending request to transmitter
print("Computer: Sending request to transmitter...")
input_data = {"message": message, "freq in": f_in, "freq out": f_out}
tx.send_json(input_data)
print("Computer: Request sent.")

# Receiving response from transmitter
print("Computer: Waiting for response from transmitter...")
tx_data = tx.recv_pyobj()
print("Computer: Response received.")

t = tx_data['time']
tx_qpsk = tx_data['qpsk']
tx_vert_lines = tx_data['vertical lines']
tx_symbols = tx_data['symbols']
message_bin = tx_data['message in binary']

# ---------------------------------------------------

# Send a request to the repeater to start listening, the transmitter to send the QPSK signal to the repeater, and the repeater to modulate the signal 

# Sending request to the repeater to start listening
print("Computer: Sending request to repeater...")
req = "Start listening."
rep.send_string(req)
print("Computer: Request sent.")
print("Computer: Waiting for response from repeater...")
rep.recv_string()
print("Computer: Response received")

# Sending request to transmitter to start sending the signal
print("Computer: Sending request to transmitter...")
req = "Transmit to repeater."
tx.send_string(req)
print("Computer: Request sent.")
print("Computer: Waiting for response from transmitter...")
tx.recv_string()
print("Computer: Response received")

# Sending request to the repeater to start modulating the signal
print("Computer: Sending request to repeater...")
rep_input = {"freq in": f_in, "freq out": f_out}
rep.send_json(rep_input)
print("Computer: Request sent.")

# Receiving data from the repeater
print("Computer: Waiting for response from repeater...")
rep_data = rep.recv_pyobj()
print("Computer: Response received.")

# ---------------------------------------------------

# Send a request to the receiver to start listening, the repeater to send the signal to the receiver, and the receiver to demodulate/decode the signal

# Sending request to the receiver to start listening
print("Computer: Sending request to receiver...")
req = "Start listening."
rx.send_string(req)
print("Computer: Request sent.")
print("Computer: Waiting for response from receiver...")
rx.recv_string()
print("Computer: Response received")

# Sending request to repeater to start sending the signal
print("Computer: Sending request to repeater...")
req = "Transmit to receiver."
rep.send_string(req)
print("Computer: Request sent.")
print("Computer: Waiting for response from repeater...")
rep.recv_string()
print("Computer: Response received")

# Sending request to the receiver to start demodulating/decoding the signal
print("Computer: Sending request to receiver...")
rx_input = {"freq out": f_out}
rx.send_json(rep_input)
print("Computer: Request sent.")

# Receiving data from the receiver
print("Computer: Waiting for response from receiver...")
rx_data = rx.recv_pyobj()
print("Computer: Response received.")

bit_sequence = rx_data['bit sequence']
recovered_message = rx_data['recovered message']

print('Computer: Original Message:', message)
print('Computer: Recovered Message:', recovered_message)


tx.close()
rep.close()
rx.close()
context.term()

