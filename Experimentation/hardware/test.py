import transmit_processing as transmit_processing
import receive_processing as receive_processing 
import numpy as np
from config import *
import matplotlib.pyplot as plt

message = input('Enter your message: \n')
sps = 20
sample_rate = 2.88e6
beta = BETA
N = NUMTAPS

tp = transmit_processing.transmit_processing(sps, sample_rate)
rp = receive_processing.receive_processing(sps, sample_rate)

bits_out, data = tp.work(message, beta, N)

bits_in, decoded_message, symbols = rp.work(data, beta, N)

print("Message Sent: ", message)
print("Message Received: ", decoded_message)
print("Bits Sent: ", bits_out)
print("Bits Received: ", bits_in)



