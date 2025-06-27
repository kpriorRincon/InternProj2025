from rtlsdr import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import asyncio
import numpy as np


# async def streaming():
#     sdr = RtlSdr()

#     sdr.sample_rate = 2.048e6  # Hz
#     sdr.center_freq = 88.1e6     # Hz
#     sdr.freq_correction = 60   # PPM
#     sdr.gain = 'auto'

#     async for samples in sdr.stream():
#         x = 0.5 * np.angle(samples[0:-1] * np.conj(samples[1:]))

#     # to stop streaming:
#     await sdr.stop()

#     # done
#     sdr.close()

# loop = asyncio.get_event_loop()
# loop.run_until_complete(streaming())

# sdr = RtlSdr()

# # configure device
# sdr.sample_rate = 2.048e6  # Hz
# sdr.center_freq = 10e6     # Hz
# sdr.freq_correction = 60   # PPM
# sdr.gain = 'auto'

# fig = plt.figure()
# graph_out = fig.add_subplot(1, 1, 1)


# def animate(i):
#     graph_out.clear()
#     #samples = sdr.read_samples(256*1024)
#     samples = sdr.read_samples(128*1024)
#     # use matplotlib to estimate and plot the PSD
#     graph_out.psd(samples, NFFT=1024, Fs=sdr.sample_rate /
#                   1e6, Fc=sdr.center_freq/1e6)


# try:
#     ani = animation.FuncAnimation(fig, animate, interval=10)
#     # plt.xlabel("Frequency (Hz)")
#     # plt.ylabel("Magnitude (dB)")
#     # plt.title("Power Spectral Density")
#     plt.show()
# except KeyboardInterrupt:
#     pass
# finally:
#     sdr.close()
import scipy.signal as signal
import time 

# configure RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6 # Hz
sdr.center_freq = 11e6 # Hz
sdr.freq_correction = 60 # PPM
sdr.gain = 'auto'
N = 1024

# # sleep
# time.sleep(1)

# # plot settings - MODIFIED
# def update_plot(data):
#     f, t, Sxx = signal.spectrogram(data, fs=sdr.sample_rate)
#     # Convert frequency bins to actual frequencies centered around SDR center frequency
#     f_actual = f - sdr.sample_rate/2 + sdr.center_freq
#     return f_actual, t, Sxx

# # Initialize plot with proper frequency scaling
# f, t, Sxx_init = update_plot(np.random.randn(N) * 0.01)
# ax = plt.subplot(1,1,1)
# im = ax.imshow(10*np.log10(Sxx_init), aspect='auto', origin='lower', 
#                extent=[t.min(), t.max(), f.min()/1e6, f.max()/1e6])  # Convert to MHz
# plt.colorbar(im, label='Power (dB)')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (MHz)')
# plt.title(f'Spectrogram - Center: {sdr.center_freq/1e6:.1f} MHz')
# plt.ion()

# # run detection
# count = 0
# while True:
#     count += 1
    
#     # read samples from RTL-SDR
#     samples = sdr.read_samples(N)
    
#     # plot samples - MODIFIED
#     f, t, Sxx = update_plot(samples)
#     im.set_data(10*np.log10(Sxx))
#     im.set_extent([t.min(), t.max(), f.min()/1e6, f.max()/1e6])  # Update extent
#     plt.draw()
#     plt.pause(0.01)

x = sdr.read_samples(N)
fft_size = 1024
num_rows = len(x) // fft_size # // is an integer division which rounds down
spectrogram = np.zeros((num_rows, fft_size))
for i in range(num_rows):
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)

plt.imshow(spectrogram, aspect='auto', extent = [sdr.sample_rate/-2/1e6, sdr.sample_rate/2/1e6, len(x)/sdr.sample_rate, 0])
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
plt.show()

sdr.close()