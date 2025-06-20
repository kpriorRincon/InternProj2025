import numpy as np
import matplotlib.pyplot as plt
from Sig_Gen import SigGen

freq_offset = 20e3
fs=1e6
def frequency_offset(qpsk_wave, t):
    return qpsk_wave * np.exp(1j * 2 * np.pi * (freq_offset) * t)

def coarse_freq_recovery(qpsk_wave, order=4):
    fft_vals_bef = np.fft.fftshift(np.abs(np.fft.fft(qpsk_wave)))
    mag_inc = 20 * np.log(np.abs(fft_vals_bef))

    freqs = np.linspace(-fs/2, fs/2, len(fft_vals_bef))
    plt.plot(freqs, qpsk_wave)
    plt.show()
    qpsk_wave_r = qpsk_wave**4

    fft_vals = np.fft.fftshift(np.abs(np.fft.fft(qpsk_wave_r)))
    plt.plot(freqs, fft_vals)
    plt.show()

    freq_tone = freqs[np.argmax(fft_vals)] / order 
    print(freq_tone)
    
    t = np.arange(len(qpsk_wave)) / fs
    fixed_qpsk = qpsk_wave * np.exp(-1j*2*np.pi*freq_tone*t)

    fft_vals_cor = np.fft.fftshift(np.abs(np.fft.fft(fixed_qpsk)))
    mag_out = 20 * np.log(np.abs(fft_vals_cor))
    return fixed_qpsk

def phase_detector_4(sample):
    if sample.real > 0:
        a = 1.0
    else:
        a = -1.0
    if sample.imag > 0:
        b = 1.0
    else:
        b = -1.0
    return a * sample.imag - b * sample.real

def costas_loop(qpsk_wave):
    # requires downconversion to baseband first
    N = len(qpsk_wave)
    phase = 0
    freq = 0 # derivative of phase; rate of change of phase
    #Following params determine feedback loop speed
    alpha = 0.3 #0.132 immediate phase correction based on current error
    beta = 0.01 #0.00932  tracks accumalated phase error
    out = np.zeros(N, dtype=np.complex64)
    freq_log = []
    for i in range(N):
        out[i] = qpsk_wave[i] * np.exp(-1j*phase) #adjust input sample by inv of estimated phase offset
        error = phase_detector_4(out[i])

        freq += (beta * error)
        freq_log.append(freq * fs / (2 * np.pi))
        phase += freq + (alpha * error)

        while phase >= 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi
    print(freq_log[-1])
    
    plt.plot(freq_log,'.-')
    plt.show()
    
def main():
    sig_gen = SigGen(freq=900e6, amp=1,sample_rate=fs, symbol_rate=1000)
    bits = sig_gen.message_to_bits('hello there'*3)
    t, qpsk_wave, _, _, _ = sig_gen.generate_qpsk(bits, False)
    qpsk_wave *= np.exp(-1j * 2 * np.pi * 900e6 * t) #baseband
    offset_qpsk = frequency_offset(qpsk_wave, t)
    coarse_fixed = coarse_freq_recovery(offset_qpsk)
    costas_loop(coarse_fixed)

if __name__ == "__main__":
    main()