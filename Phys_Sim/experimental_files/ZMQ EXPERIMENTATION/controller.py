# Scipts for a ZeroMQ system that implments the bent-pipe communication
# This script acts as the controller/computer that sends commands to the different nodes

import zmq
import time

context = zmq.Context()

# Controller sockets
tx = context.socket(zmq.REQ)
tx.connect("tcp://127.0.0.1:6001") # sets up a REQ-REP connection with the transmitter

rep = context.socket(zmq.REQ)
rep.connect("tcp://127.0.0.1:6002") # sets up a REQ-REP connection with the repeater

rx = context.socket(zmq.REQ)
rx.connect("tcp://127.0.0.1:6003") # sets up a REQ-REP connection with the receiver