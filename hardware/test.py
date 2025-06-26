import transmit_processing as transmit_processing
import receive_processing as receive_processing 

message = input('Enter your message: \n')
sps = 4
sample_rate = 2.88e6
beta = 0.35
N = 40

transmit_processing = transmit_processing.transmit_processing(sps, sample_rate)
receive_processing = receive_processing.receive_processing(sps, sample_rate)

bits_out, data = transmit_processing.work(message, beta, N)

bits_in, decoded_message = receive_processing.work(data, beta, N)

print("Message Sent: ", message)
print("Message Received: ", decoded_message)
print("Bits Sent: ", bits_out)
print("Bits Received: ", bits_in)
