from Sig_Gen import SigGen
import numpy as np



fs = 2.88e6
symb_rate = 1e6/20
freq_offset = 20e3

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


def coarse_freq_recovery(qpsk_wave, order=4):

    qpsk_wave_r = qpsk_wave**4

    fft_vals = np.fft.fftshift(np.abs(np.fft.fft(qpsk_wave_r)))
    freqs = np.linspace(-fs/2, fs/2, len(fft_vals))

    freq_tone = freqs[np.argmax(fft_vals)] / order 
    
    t = np.arange(len(qpsk_wave)) / fs
    fixed_qpsk = qpsk_wave * np.exp(-1j*2*np.pi*freq_tone*t)
    
    return fixed_qpsk, freq_tone


def costas_loop(qpsk_wave, alpha, beta):
    # requires downconversion to baseband first
    N = len(qpsk_wave)
    phase = 0
    freq = 0 # derivative of phase; rate of change of phase (radians/sample)
    #Following params determine feedback loop speed
    #alpha = 0.002#0.0006 #0.132 immediate phase correction based on current error
    #beta = 0.000000634#0.0000004 #0.00932  tracks accumalated phase error
    out = np.zeros(N, dtype=np.complex64)
    freq_log = []
    
    for i in range(N):
        out[i] = qpsk_wave[i] * np.exp(-1j*phase) #adjust input sample by inv of estimated phase offset
        error = phase_detector_4(out[i])

        freq += (beta * error)
        #log frequency in Hz
        freq_log.append(freq * fs / (2 * np.pi))
        phase += freq + (alpha * error)

        while phase >= 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi
    
    return freq_log[-1]



def main():

    #Generate QPSK at carrier frequency
    sig_gen = SigGen(freq=900e6, amp=1,sample_rate=fs, symbol_rate=symb_rate)
    bits = sig_gen.message_to_bits('hello there' * 3)
    t, qpsk_wave = sig_gen.generate_qpsk(bits)

    # Set frequency Offset
    qpsk_wave = qpsk_wave * np.exp(1j* 2 * np.pi * freq_offset * t) # shifts down by freq offset
    
    #Tune down to baseband
    tuned_sig = qpsk_wave * np.exp(-1j * 2 * np.pi * sig_gen.freq * t)
    coarse_fixed_sig, coarse_freq = coarse_freq_recovery(tuned_sig)

    print(f"Coarse Frequency Correction: {coarse_freq} Hz")

    best_alpha = None
    best_beta = None
    min_error = float('inf')
    best_correction = None

    alphas = np.arange(0, 0.2, 0.001)
    betas = np.arange(0, 0.01, 0.00001)

    print(f"Testing {len(alphas)} alphas and {len(betas)} betas.")
    print(f"Total number of test cases: {len(alphas) * len(betas)}.")

    counter = 0
    for a in alphas:
        for b in betas:
            fine_freq = costas_loop(coarse_fixed_sig, alpha=a, beta=b)
            total_correction = coarse_freq + fine_freq
            error = abs(freq_offset - total_correction)
            #compare to freq_offset
            print(f"Test Case: {counter}", end='\r')
            counter += 1
            if error < min_error:
                min_error = error
                best_alpha = a
                best_beta = b
                best_correction = fine_freq

    print(f"Best alpha: {best_alpha}")
    print(f"Best beta: {best_beta}")
    print(f"Best Costas Correction {best_correction} Hz")
    print(f"Total correction: {best_correction + coarse_freq} Hz")
    print(f"Minimum error: {min_error} Hz")

if __name__ == "__main__":
    main()
