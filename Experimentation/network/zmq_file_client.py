# zmq_recv_dict_client.py
import zmq
import json

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://10.232.62.2:5555")  # Replace with server IP

socket.send_string("GET_DATA")
msg = socket.recv_string()

data = json.loads(msg)
print("Received dictionary:", data)
