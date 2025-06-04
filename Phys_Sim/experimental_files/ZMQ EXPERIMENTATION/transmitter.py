import zmq
import time

context = zmq.Context() 

# Control socket
ctrl = context.socket(zmq.REP)
ctrl.bind("tcp://127.0.0.1:6001") # sets up a REQ-REP connection with the controller/computer

# Data socket
data = context.socket(zmq.PUSH)
data.bind("tcp://127.0.0.1:3001") # sets up a PUSH-PULL connection with the repeater
