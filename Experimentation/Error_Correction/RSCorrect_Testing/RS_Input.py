import RS_Transmit_Processing as RS_Transmit_Processing
import RS_Receive_Processing as RS_Receive_Processing 
import numpy as np
from config import *
import reedsolo as rs
import time

# Initialize RS codec with GF(2^8) primitive polynomial
rs.init_tables(0x11d)

# get user input
message = input('Enter your message: \n')

# initialize tx and rx parameters
sps = 20
sample_rate = 2.88e6
beta = BETA
N = NUMTAPS

# call tx and rx objects
tp = RS_Transmit_Processing.transmit_processing(sps, sample_rate)
rp = RS_Receive_Processing.receive_processing(sps, sample_rate)

# convert message to bits for transmission
bits_out, data = tp.work(message, beta, N)
print("Data to send: ", bits_out)

# decode the received data
bits_in, decoded_message, symbols = rp.work(data, beta, N)
byte_data = int(decoded_message, 2).to_bytes((len(decoded_message) + 7) // 8, 'big')# convert the bit string to bytes
print("Bytes: ", byte_data)

# error check
timer = time.time()
check = calculator.checksum(byte_data)
print("Time to run CRC check: ", time.time() - timer)
print("Remainder: ", check)
if check == 0:
    print("Data is valid...")
    # print the results
    print("Message Sent: ", message)
    print("Message Received: ", byte_data[:-1].decode('utf-8'))
    print("Bits Sent: ", bits_out)
    print("Bits Received: ", bits_in)
else:
    print("Message Received: ", byte_data.decode('utf-8'))
    print("Data is invalid...\nAborting...")
