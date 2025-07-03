import receive_processing as receive_processing 
import numpy as np
import matplotlib.pyplot as plt
 
sps = 20
sample_rate = 2.88e6
beta = 0.35
N = 101
  
rp = receive_processing.receive_processing(sps, sample_rate)
  
start = 7920
end = 11439

data = np.fromfile("selected_signal.bin", dtype=np.complex64)

bits_in, decoded_message, symbols = rp.work(data, beta, N)
  
plt.scatter(np.real(symbols), np.imag(symbols), s=10, c='b')
plt.ylabel("Quadrature (Q)")
plt.xlabel("In-Phase (I)")
plt.title("Received Symbols")
plt.show()

print("Message Received: ", decoded_message)
print("Bits Received: ", bits_in)
  
