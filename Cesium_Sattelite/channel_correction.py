import numpy as np
import matplotlib.pyplot as plt
from Sig_Gen import SigGen
from scipy import signal

freq_offset = 20e3
fs=1e6


def fractional_delay(t, signal, delay_in_sec, Fs):
    """
    Apply fractional delya using interpolation (sinc-based)
    'signal' a signal to apply the time delay
    'delay_in_seconds' delay to apply to signal
    'Fs' Sample rate used to convert delay to units of samples
    """
    #first step we need to shift by integer 
    #for now we are only gonna get the fractional part of the delay
    total_delay = delay_in_sec * Fs # = samples
    fractional_delay = total_delay % 1 # to get the remainder
    integer_delay = int(total_delay)

    print(f'fractional delay in samples: {fractional_delay}')
    #then shift by fractional samples with the leftover 
    delay_in_samples = fractional_delay
    #Fs * delay_in_sec # samples/seconds * seconds = samples
    
    #filter taps
    N = 301
    #construct filter
    n = np.arange(-N//2, N//2)
    h = np.sinc(n-delay_in_samples)
    h *= np.hamming(N) #something like a rectangular window
    h /= np.sum(h) #normalize to get unity gain, we don't want to change the amplitude/power

    #apply filter: keep original length and center delay
    new_signal = np.convolve(signal, h, mode='same')

    #cut out Group delay from FIR filter
    # delay = (N - 1) // 2 # group delay of FIR filter is always (N - 1) / 2 samples, N is filter length (of taps)
    # padded_signal = np.pad(new_signal, (0, delay), mode='constant')
    # new_signal = padded_signal[delay:]  # Shift back by delay

    #create a new time vector with the correcct size
    new_t = t[0] + np.arange(len(new_signal)) / Fs # t[0] might be 0 but if it isn't the rest of the time vector will be offset by this amount
    
    return new_t, new_signal


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

def costas_loop(qpsk_wave, t):
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
    # plt.show()
    return np.exp(-1j*2* np.pi * freq * t)
    
def mueller(samples, sps):
    samples_interpolated = signal.resample_poly(samples, 16, 1)

    mu = 0 # initial estimate of phase of sample
    out = np.zeros(len(samples) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex64) # stores values, each iteration we need the previous 2 values plus current value
    i_in = 0 # input samples index
    i_out = 2 # output index (let first two outputs be 0)
    
    while i_out < len(samples) and i_in+16 < len(samples):
        out[i_out] = samples_interpolated[i_in*16 + int(mu*16)]        
        out_rail[i_out] = int(np.real(out[i_out]) > 0) + 1j*int(np.imag(out[i_out]) > 0)
        x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
        y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
        mm_val = np.real(y - x)
        mu += sps + 0.3*mm_val
        i_in += int(np.floor(mu)) # round down to nearest int since we are using it as an index
        mu = mu - np.floor(mu) # remove the integer part of mu
        i_out += 1 # increment output index

    out = out[2:i_out] # remove the first two, and anything after i_out (that was never filled out)
    samples = out # only include this line if you want to connect this code snippet with the Costas Loop later on
    return samples # the output signal


def main():
    sig_gen = SigGen(freq=900e6, amp=1,sample_rate=fs, symbol_rate=1000)
    bits = sig_gen.message_to_bits('hello there'*3)
    print(f'num symbols: {len(bits)//2}')
    t, qpsk_wave = sig_gen.generate_qpsk(bits)
    print(len(qpsk_wave))
    #test time correction
    new_t, new_signal = fractional_delay(t, qpsk_wave, 0.4023042, fs)
    time_corrected = mueller(new_signal, fs/1000)
    """
    plt.subplot(2, 1, 1)
    plt.plot(np.real(time_corrected), label = "Real")
    plt.subplot(2,1,2)
    plt.plot(np.imag(time_corrected), label = "Imag")
    plt.figure(figsize=(10, 5))
    """

    plt.subplot(2,2,1)
    plt.title("Constellation: orig")
    plt.scatter(np.real(qpsk_wave), np.imag(qpsk_wave), s=2, alpha=0.5)
    idx = np.arange(0, len(qpsk_wave), 1000)
    plt.scatter(np.real(qpsk_wave)[idx], np.imag(qpsk_wave)[idx], color='red', s=20, label='Every 1000th')
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.legend()
    
    plt.subplot(2,2,2)
    plt.title("Constellation: new_signal")
    plt.scatter(np.real(new_signal), np.imag(new_signal), s=2, alpha=0.5)
    idx = np.arange(0, len(new_signal), 1000)
    plt.scatter(np.real(new_signal)[idx], np.imag(new_signal)[idx], color='red', s=20, label='Every 1000th')
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.legend()
    
    plt.subplot(2,2,3)
    plt.title("Constellation: time_corrected")
    plt.scatter(np.real(time_corrected), np.imag(time_corrected), s=2, alpha=0.5)
    idx = np.arange(0, len(time_corrected), 1000)
    plt.scatter(np.real(time_corrected)[idx], np.imag(time_corrected)[idx], color='red', s=20, label='Every 1000th')
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.legend()
    plt.show()
    # qpsk_wave *= np.exp(-1j * 2 * np.pi * 900e6 * t) #baseband
    # offset_qpsk = frequency_offset(qpsk_wave, t)
    # coarse_fixed = coarse_freq_recovery(offset_qpsk)
    # costas_loop(coarse_fixed)

if __name__ == "__main__":
    main()