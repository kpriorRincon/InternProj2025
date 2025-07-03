#goal find auto correlation of start signal 

from scipy.signal import fftconvolve
import transmit_processing as tp

transmitter = tp.transmit_processing(sps=4, sample_rate=1000)
start_sequence, end_sequence = transmitter.generate_markers()

#modulate them
start_sequence, end_sequence = transmitter.modulated_markers(beta=0.35, N=41, start_sequence = start_sequence, end_sequence = end_sequence)
import numpy as np
start_auto = fftconvolve(start_sequence ** 4, np.conj(np.flip(start_sequence))**4, mode = 'same')
end_auto = fftconvolve(end_sequence**4, np.conj(np.flip(end_sequence))**4, mode = 'same')
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))

plt.plot(np.abs(start_auto), label='Start Auto-correlation')
plt.title('Auto-correlation of Start Marker')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid()

plt.figure(figsize=(10, 5))

plt.plot(np.abs(end_auto), label='End Auto-correlation')
plt.title('Auto-correlation End')

plt.show()