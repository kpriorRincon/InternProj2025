import hardware.CRC_Testing.CRC_Transmit_Processing as CRC_Transmit_Processing
import receive_processing as receive_processing 
import numpy as np
from config import *
from crc import Calculator, Crc8

# initialize the CRC calculator
calculator = Calculator(Crc8.CCITT)

# get user input
message = input('Enter your message: \n')

# initialize tx and rx parameters
sps = 20
sample_rate = 2.88e6
beta = BETA
N = NUMTAPS

# call tx and rx objects
tp = CRC_Transmit_Processing.transmit_processing(sps, sample_rate)
rp = receive_processing.receive_processing(sps, sample_rate)

# convert message to bits for transmission
bits_out, data = tp.work(message, beta, N)

# decode the received data
bits_in, decoded_message, symbols = rp.work(data, beta, N)

# error check
check = calculator.checksum(data)
print("Remainder: ", check)
if check == 0:
    print("Data is valid...")
    # print the results
    print("Message Sent: ", message)
    print("Message Received: ", decoded_message)
    print("Bits Sent: ", bits_out)
    print("Bits Received: ", bits_in)
else:
    print("Data is invalid...\nAborting...")

#start_marker, end_marker = tp.generate_markers()
#start_data, end_data = tp.modulated_markers(beta, N, start_marker, end_marker)

#zeros = np.zeros(len(start_data)*3, dtype=np.complex64)
#data = np.append(start_data, zeros)

#data.tofile("data_for_sighound.bin")

