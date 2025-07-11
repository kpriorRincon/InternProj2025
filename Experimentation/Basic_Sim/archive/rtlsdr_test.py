from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt

def main():
    sdr = RtlSdr()
    # Configure the SDR
    sdr.sample_rate = 2.048e6  # Hz
    sdr.center_freq = 100e6  # Hz
    sdr.gain = 'auto'  # Automatic gain control
    # Read samples
    samples = sdr.read_samples(256*1024)  # Read 256k samples
    # Convert samples to complex numbers
    sdr.close()
    psd =np.abs(np.fft.fft(samples))**2/len(samples)
    psd_shifted = np.fft.fftshift(psd)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sdr.sample_rate))
    np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sdr.sample_rate))

   # Plotting
    plt.plot(freqs/1e6, 10*np.log10(psd_shifted)) # Frequency in MHz
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.title("Spectrum")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()