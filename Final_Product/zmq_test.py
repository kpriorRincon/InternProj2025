import zmq
#port 5555

context = zmq.Context()
Rx = context.socket(zmq.REQ)
Rx.connect('tcp://10.232.62.2:5555') # connect to server

Rx.send_string("Hello from client")
reply = Rx.recv_string()
print(f'received reply: {reply}')