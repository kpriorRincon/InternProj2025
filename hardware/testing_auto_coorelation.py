import numpy as np
import matplotlib.pyplot as plt
#goal find auto correlation of start signal 

from scipy.signal import fftconvolve
import transmit_processing as tp

transmitter = tp.transmit_processing(sps=20, sample_rate=2.88e6)

#modulate them
start_sequence, end_sequence = transmitter.modulated_markers(beta=0.35, N=41)
start_auto = np.abs(fftconvolve(start_sequence, np.conj(np.flip(start_sequence)), mode = 'same'))
print(f'start_sequence = {len(start_sequence)}, start_auto = {len(start_auto)}')
plt.figure(figsize=(10,6))
plt.plot(start_auto)
plt.xlabel('Sample Index')
plt.title('Auto-correlation of Start Marker')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# # Plot middle samples of start_sequence
# mid = len(start_sequence) // 2
# window = 50  # number of samples to show around the center
# start_slice = slice(mid - window, mid + window)
# plt.figure(figsize=(10, 4))
# plt.plot(np.real(start_sequence[start_slice]), label='Real')
# plt.plot(np.imag(start_sequence[start_slice]), label='Imag')
# plt.title('Start Sequence (Middle Samples)')
# plt.legend()
# plt.show()

# # Plot middle samples of end_sequence
# mid = len(end_sequence) // 2
# end_slice = slice(mid - window, mid + window)
# plt.figure(figsize=(10, 4))
# plt.plot(np.real(end_sequence[end_slice]), label='Real')
# plt.plot(np.imag(end_sequence[end_slice]), label='Imag')
# plt.title('End Sequence (Middle Samples)')
# plt.legend()
# plt.show()


# start_auto = fftconvolve(start_sequence, np.conj(np.flip(start_sequence)), mode = 'same')
# end_auto = fftconvolve(end_sequence, np.conj(np.flip(end_sequence)), mode = 'same')
# plt.figure(figsize=(10, 5))

# plt.plot(np.abs(start_auto), label='Start Auto-correlation')
# plt.title('Auto-correlation of Start Marker')
# plt.xlabel('Sample Index')
# plt.ylabel('Amplitude')
# plt.grid()

# plt.figure(figsize=(10, 5))

# plt.plot(np.abs(end_auto), label='End Auto-correlation')
# plt.title('Auto-correlation End')
# plt.show()




# normalized_start_sequence = (start_sequence - np.min(start_sequence)) / (np.max(np.abs(start_sequence)) - np.min(start_sequence))
# normalized_end_sequence = (end_sequence - np.min(end_sequence)) / (np.max(np.abs(end_sequence)) - np.min(end_sequence))

# start_auto_normalized = fftconvolve(normalized_start_sequence, np.conj(np.flip(start_sequence)), mode = 'same')
# end_auto_normalized = fftconvolve(normalized_end_sequence, np.conj(np.flip(end_sequence)), mode = 'same')




# # Plot middle samples of end_sequence
# mid = len(normalized_end_sequence) // 2
# end_slice = slice(mid - window, mid + window)
# plt.figure(figsize=(10, 4))
# plt.plot(np.real(normalized_end_sequence[end_slice]), label='Real')
# plt.plot(np.imag(normalized_end_sequence[end_slice]), label='Imag')
# plt.title('End Sequence (Middle Samples) Normalized')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 5))

# plt.plot(np.abs(start_auto_normalized), label='Start Auto-correlation')
# plt.title('Normalized Start correlated with start')

# plt.figure(figsize=(10, 5))

# plt.plot(np.abs(end_auto_normalized), label='End Auto-correlation')
# plt.title('Normalized End correlated with end')
# plt.show()



# #testing a different normalization method
# normalized_start_sequence /= np.sqrt(np.mean(np.abs(start_sequence)**2))
# normalized_end_sequence /= np.sqrt(np.mean(np.abs(end_sequence)**2))

# # Plot middle samples of start_sequence
# mid = len(normalized_start_sequence) // 2
# window = 50  # number of samples to show around the center
# start_slice = slice(mid - window, mid + window)
# plt.figure(figsize=(10, 4))
# plt.plot(np.real(normalized_start_sequence[start_slice]), label='Real')
# plt.plot(np.imag(normalized_start_sequence[start_slice]), label='Imag')
# plt.title('Start Sequence (Middle Samples) Normalized(ALTERNATIVE)')
# plt.legend()
# plt.show()

# # Plot middle samples of end_sequence
# mid = len(normalized_end_sequence) // 2
# end_slice = slice(mid - window, mid + window)
# plt.figure(figsize=(10, 4))
# plt.plot(np.real(normalized_end_sequence[end_slice]), label='Real')
# plt.plot(np.imag(normalized_end_sequence[end_slice]), label='Imag')
# plt.title('End Sequence (Middle Samples) Normalized(ALTERNATIVE')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 5))

# plt.plot(np.abs(start_auto_normalized), label='Start Auto-correlation')
# plt.title('Normalized (ALTERNATIVE) Start correlated with start')

# plt.figure(figsize=(10, 5))

# plt.plot(np.abs(end_auto_normalized), label='End Auto-correlation')
# plt.title('Normalized (ALTERNATIVE)End correlated with end')
# plt.show()