import numpy as np
import matplotlib.pyplot as plt

#testing
symbol_rate = 10e6 #10M symbols per sec
f_carrier = 910e6 #10e3
fs = 4e9 #100e3 #sample frequency
#duration = 0.01
desired_f = 1e9#15e3
#A QPSK signal contains two main components: in-phase and quadrature
def generate_qpsk_wave(bitstream):
    """
    - bitstream: A list of bits to be transmitted

    Returns:
    - t: Time vector for plotting
    - qpsk_signal: The generated QPSK signal (complex)
    """
    # Generate the time vector
    symbol_duration = 1 / symbol_rate
    duration = symbol_duration * (len(bitstream) // 2)
    samples_per_symbol = int(fs * symbol_duration)  # Number of samples per symbol
    total_samples = int(fs * duration)  # Total number of samples

    t = np.linspace(0, duration, total_samples, endpoint=False)

    qpsk_signal = np.zeros(total_samples, dtype=complex)

    for i in range(0, len(bitstream), 2):
        b1, b2 = bitstream[i], bitstream[i+1]
        
        if b1 == 0 and b2 == 0:
            I_val, Q_val = 1, 1
        elif b1 == 0 and b2 == 1:
            I_val, Q_val = 1, -1
        elif b1 == 1 and b2 == 0:
            I_val, Q_val = -1, 1
        elif b1 == 1 and b2 == 1:
            I_val, Q_val = -1, -1

        symbol = complex(I_val, Q_val)
        phase_rad = np.angle(symbol)
        phase_deg = np.degrees(phase_rad)

        print(f"Bits: {b1}{b2}, Symbol: {symbol}, Phase: {phase_deg:.2f}°")
        # Create the symbol time window
        start_index = (i // 2) * samples_per_symbol
        end_index = start_index + samples_per_symbol
        
        symbol_t = t[start_index:end_index]

        I = I_val * np.cos(2 * np.pi * f_carrier * symbol_t)
        Q = Q_val * np.sin(2 * np.pi * f_carrier * symbol_t)

        qpsk_signal[start_index:end_index] = I + Q

    print(qpsk_signal)
    return t, qpsk_signal
def mixing(t, input_qpsk, desired_f, f_carrier):
    mixing_signal = np.cos(2 * np.pi * (desired_f + f_carrier) * t)
    # Mix the QPSK signal with the complex exponential to shift its frequency
    qpsk_shifted = input_qpsk * mixing_signal

    return qpsk_shifted

def plotting(t, input_qpsk, qpsk_shifted):
    # Compute FFT
    n = len(t)
    freqs = np.fft.fftfreq(n, d=1/fs)

    # FFT of original and shifted signals
    fft_input = np.fft.fft(input_qpsk)
    fft_shifted = np.fft.fft(qpsk_shifted)

    # Convert magnitude to dB
    mag_input = 20 * np.log10(np.abs(fft_input))
    mag_shifted = 20 * np.log10(np.abs(fft_shifted))

    plt.figure(figsize=(12, 10))

    # --- Time-domain plot: Original QPSK ---
    plt.subplot(2, 2, 1)
    plt.plot(t, np.real(input_qpsk))  # convert time to microseconds
    plt.title("Original QPSK Signal (Time Domain)")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1e-7)
    plt.grid(True)

    # --- Time-domain plot: Shifted QPSK ---
    plt.subplot(2, 2, 2)
    plt.plot(t, np.real(qpsk_shifted))
    plt.title("Shifted QPSK Signal (Time Domain)")
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1e-7)
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(freqs, mag_input, label="Original QPSK", alpha=0.8)
    #plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT of QPSK Before and After Frequency Shift")
    plt.xlim(0, fs / 2)  # From 0 to fs in MHz
    plt.ylim(0, np.max(mag_input) + 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT of QPSK Before and After Frequency Shift")
    plt.xlim(0, fs / 2)  # From 0 to fs in MHz
    plt.ylim(0, np.max(mag_input) + 10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    bitstream = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1]

    t, input_qpsk = generate_qpsk_wave(bitstream)

    #ideal mixing_signal = np.exp(1j * 2 * np.pi * (desired_f - f_carrier) * t)

    qpsk_shifted = mixing(t, input_qpsk, desired_f, f_carrier)

    plotting(t, input_qpsk, qpsk_shifted)
    
if __name__ == "__main__":
    main()