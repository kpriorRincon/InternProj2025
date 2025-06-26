from rtlsdr import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import asyncio
import numpy as np


async def streaming():
    sdr = RtlSdr()

    sdr.sample_rate = 2.048e6  # Hz
    sdr.center_freq = 88.1e6     # Hz
    sdr.freq_correction = 60   # PPM
    sdr.gain = 'auto'

    async for samples in sdr.stream():
        x = 0.5 * np.angle(samples[0:-1] * np.conj(samples[1:]))

    # to stop streaming:
    await sdr.stop()

    # done
    sdr.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(streaming())

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