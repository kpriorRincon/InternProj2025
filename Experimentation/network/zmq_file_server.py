# zmq_file_server.py
import zmq
import json

FILE_PATH = "myfile.txt"  # File to send
PORT = 5555

data = {
    'Name': 'Jorge',
    'Device #': 'ab07',
    'status': True
}

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://0.0.0.0:{PORT}")

print(f"ZMQ server ready. Waiting for request on port {PORT}...")

while True:
    message = socket.recv_string()
    print(f"Received request: {message}")

    if message == "GET_DATA":
        socket.send_string(json.dumps(data))
        print("Data sent.")
    else:
        socket.send_string("Unknown request.")
