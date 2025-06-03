import zmq
import time

context = zmq.Context()


# Control socket
ctrl = context.socket(zmq.REP)
ctrl.bind("tcp://127.0.0.1:6002") # sets up a REQ-REP connection with the controller/computer

# Data sockets
data = context.socket(zmq.PULL)
data.connect("tcp://127.0.0.1:3001") # sets up a PUSH-PULL connection with the transmitter

data = context.socket(zmq.PUSH)
data.bind("tcp://127.0.0.1:3002") # sets up a PUSH-PULL connection with the receiver
