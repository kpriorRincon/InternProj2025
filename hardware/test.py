import transmit_processing as transmit_processing
import receive_processing as receive_processing 
import numpy as np

message = input('Enter your message: \n')
sps = 20
sample_rate = 2.88e6
beta = 0.35
N = 101

tp = transmit_processing.transmit_processing(sps, sample_rate)
rp = receive_processing.receive_processing(sps, sample_rate)

bits_out, data = tp.work(message, beta, N)

bits_in, decoded_message, symbols = rp.work(data, beta, N)

print("Message Sent: ", message)
print("Message Received: ", decoded_message)
print("Bits Sent: ", bits_out)
print("Bits Received: ", bits_in)

#start_marker, end_marker = tp.generate_markers()
#start_data, end_data = tp.modulated_markers(beta, N, start_marker, end_marker)

#zeros = np.zeros(len(start_data)*3, dtype=np.complex64)
#data = np.append(start_data, zeros)

#data.tofile("data_for_sighound.bin")

