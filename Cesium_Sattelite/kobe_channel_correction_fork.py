import numpy as np
import matplotlib.pyplot as plt
from Sig_Gen import SigGen, rrc_filter
from scipy import signal
freq_offset = 2210.123
fs = 40e6
symb_rate = 2e6


def fractional_delay(t, signal, delay_in_sec, Fs):
    """
    Apply fractional delya using interpolation (sinc-based)
    'signal' a signal to apply the time delay
    'delay_in_seconds' delay to apply to signal
    'Fs' Sample rate used to convert delay to units of samples
    """
    import numpy as np 
    #first step we need to shift by integer 
    #for now we are only gonna get the fractional part of the delay
    total_delay = delay_in_sec * Fs # seconds * samples/second = samples
    fractional_delay = total_delay % 1 # to get the remainder
    integer_delay = int(total_delay)

    print(f'fractional delay in samples: {fractional_delay}')
    #then shift by fractional samples with the leftover 
    delay_in_samples = fractional_delay
    #Fs * delay_in_sec # samples/seconds * seconds = samples
    
    #pad with zeros for the integer_delay
    if integer_delay > 0:
        signal = np.concatenate([np.zeros(integer_delay, dtype=complex), signal])
    
    #filter taps
    N = 301
    #construct filter
    n = np.arange(-N//2, N//2)
    h = np.sinc(n-delay_in_samples)
    h *= np.hamming(N) #something like a rectangular window
    h /= np.sum(h) #normalize to get unity gain, we don't want to change the amplitude/power


    #apply filter: same time-aligned output with the same size as the input
    new_signal = np.convolve(signal, h, mode='full')
    delay = (N - 1) // 2
    new_signal = new_signal[delay:delay+len(signal)]

    if len(new_signal) == len(signal):
        print('signal lengths are the same: with mode = full and trimming')

    #cut out Group delay from FIR filter
    # delay = (N - 1) // 2 # group delay of FIR filter is always (N - 1) / 2 samples, N is filter length (of taps)
    # padded_signal = np.pad(new_signal, (0, delay), mode='constant')
    # new_signal = padded_signal[delay:]  # Shift back by delay

    #create a new time vector with the correcct size
    new_t = np.arange(len(new_signal)) / Fs # t[0] might be 0 but if it isn't the rest of the time vector will be offset by this amount
    
    #debug---------------------
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(t,np.real(signal),label='OG')
    # plt.plot(new_t, np.real(new_signal), label='Delayed')
    # plt.legend()
    # plt.grid()
    # plt.show()
    #end debug-------------------

    return new_t, new_signal


def frequency_offset(qpsk_wave, t):
    return qpsk_wave * np.exp(1j * 2 * np.pi * (freq_offset) * t)

def coarse_freq_recovery(qpsk_wave, order=4):
    fft_vals_bef = np.fft.fftshift(np.abs(np.fft.fft(qpsk_wave)))
    mag_inc = 20 * np.log(np.abs(fft_vals_bef))

    freqs = np.linspace(-fs/2, fs/2, len(fft_vals_bef))
    plt.plot(freqs, fft_vals_bef)
    plt.title('fft before')
    plt.show()
    qpsk_wave_r = qpsk_wave**4

    fft_vals = np.fft.fftshift(np.abs(np.fft.fft(qpsk_wave_r)))
    plt.plot(freqs, fft_vals)
    plt.axvline(x=freqs[np.argmax(fft_vals)], color='r', linestyle=':', label=f'Peak: {freqs[np.argmax(fft_vals)]:.1f} Hz')
    plt.annotate('Divide frequency\nby 4\n(signal raised\nto 4th power)', 
                 xy=(freqs[np.argmax(fft_vals)], np.max(fft_vals)), 
                 xytext=(freqs[np.argmax(fft_vals)], np.max(fft_vals)*0.8),
                 fontsize=10, color='blue')
    plt.legend()
    plt.title('fft of sig^4 after')
    plt.show()

    freq_tone = freqs[np.argmax(fft_vals)] / order 
    print(f'frequency offset(coarse freq): {freq_tone}')
    
    t = np.arange(len(qpsk_wave)) / fs
    fixed_qpsk = qpsk_wave * np.exp(-1j*2*np.pi*freq_tone*t)

    #after coarse detection
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

def costas_loop(qpsk_wave, sps):
    # requires downconversion to baseband first
    N = len(qpsk_wave)
    phase = 0
    freq = 0 # derivative of phase; rate of change of phase (radians/sample)
    #Following params determine feedback loop speed
    alpha = 0.0001 # immediate phase correction based on current error
    beta = 0.000001 #  tracks accumalated phase error
    out = np.zeros(N, dtype = complex)
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
    
    #finds the frequency at the end when it converged
    print(f' converged frequency offset: {freq_log[-1]}')

    plt.plot(freq_log,'.-')
    plt.title('freq converge')
    plt.show()
    t = np.arange(len(qpsk_wave)) / fs
    return qpsk_wave * np.exp(-1j * 2 * np.pi * freq_log[-1] * t)
    
def mueller(samples, sps):
    #interpolate with a factor of 16 for fractional delay
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
    return samples # the output signal (will have sps = 1)


def runCorrection(signal, FS, symbol_rate):
    from scipy.signal import fftconvolve
    #1. Apply RRC to incoming signal 
    og_len = len(signal)
    _, h = rrc_filter(0.4, 301, 1/symbol_rate, FS)
    signal = np.convolve(signal, h, mode = 'full')    
    delay = (301 - 1) // 2 #account for group delay
    signal = signal[delay:delay + og_len]
    #correct the group delay
     
    #note at this point we would want to correlate with a signal that has been RC filtered 
    # not RRC filtered because it has been through 2 RRC filters at this point in the chain

    #2. Coarse Freq first?
    freq_corrected = coarse_freq_recovery(signal)

    #3 Correlation
    sig_gen = SigGen(0, 1.0, FS, symbol_rate) # if f = 0 this won't up mix so we'll get the baseband signal 
        # 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 0 0 0 1 0 1 0 1 1 1 0 1 1 0 0 0 1
    start_sequence = [1, 1, 1, 1, 1, 0, 0, 1,
                    1, 0, 1, 0, 0, 1, 0, 0,
                    0, 0, 1, 0, 1, 0, 1, 1,
                    1, 0, 1, 1, 0, 0, 0, 1]
    _, start_waveform = sig_gen.generate_qpsk(start_sequence)

    #then apply RRC to make a full RC filter to compare agianst
    og_len = len(start_waveform)
    start_waveform = np.convolve(start_waveform, h, mode = 'full')
    start_waveform = start_waveform[delay:delay + og_len]
    plt.figure()
    plt.plot(start_waveform)
    plt.title('start waveform to compare against')

    #start_waveform = np.pad(start_waveform, (0, delay), mode='constant')
    #start_waveform = start_waveform[delay:]
    # 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0 1 0 0 1 0
    end_sequence = [0, 0, 1, 0, 0, 1, 1, 0,
                    1, 0, 0, 0, 0, 0, 1, 0, 
                    0, 0, 1, 1, 1, 1, 0, 1, 
                    0, 0, 0, 1, 0, 0, 1, 0]
    
    _, end_waveform = sig_gen.generate_qpsk(end_sequence)
    og_len = len(end_waveform) 

    #then apply RRC to make a full RC filter to compare agianst
    end_waveform = np.convolve(end_waveform, h, mode = 'same')
    end_waveform = end_waveform[delay:delay+og_len] 
    plt.figure()
    plt.plot(end_waveform)
    plt.title('end waveform to compare against')
    #end_waveform = np.pad(end_waveform, (0, delay), mode='constant')
    #end_waveform = end_waveform[delay:]
    # now run cross correlation
    plt.figure()
    plt.plot(freq_corrected[0:350])
    plt.title('incoming signal (first 350 smaples)')

    plt.figure()
    plt.plot(freq_corrected[-350:])
    plt.title('incoming signal (last 350 samples)')

    #find start index by convolving signal with preamble
    start_corr_sig = fftconvolve(freq_corrected, np.conj(np.flip(start_waveform)), mode = 'same')
    plt.figure()
    plt.title('start correlation')
    plt.plot(start_corr_sig)
    end_corr_signal = fftconvolve(freq_corrected, np.conj(np.flip(end_waveform)), mode = 'same')
    plt.figure()
    plt.plot(end_corr_signal)
    plt.title('end correlation')
    plt.show()
    #get the index
    start = np.argmax(np.abs(start_corr_sig)) - int((8) * (FS/symbol_rate)) # If the preamble is 32 bits long, its 16 symbols, symbols * samples/symbol = samples
    end = np.argmax(np.abs(end_corr_signal)) + int(8 * (FS/symbol_rate))
    print(f'start: {start}, end: {end}')


    #the signal will contain the markers at the begining and the end
    print(f'length of signal: {len(signal)}\n signal: {signal}')

    signal = freq_corrected[start:end]
    print(f'length of signal after trim: {len(signal)}\n signal: {signal}')
    #testing if coarse time correction is good enough
    #down sample because costas expects decimated signal
    #signal = signal[::int(FS/symbol_rate)]


    # print(f'length of signal: {len(signal)}\n signal: {signal}')
    #4. Time Synch (Muellers)
    # mueller_corrected = mueller(signal, FS/symbol_rate)



    # #5. Fine Freq (Costas)
    signal_ready_for_demod = costas_loop(signal, FS/symbol_rate)
    
    #decimation has already happened with mueller and muller
    return signal_ready_for_demod[::int(FS/symbol_rate)]

def main():
    sig_gen = SigGen(freq=900e6, amp=1,sample_rate=fs, symbol_rate=symb_rate)
    bits = sig_gen.message_to_bits('hello there' * 3)
    print(f'num symbols: {len(bits)//2}')
    t, qpsk_wave = sig_gen.generate_qpsk(bits)
    print(f'length of wave: {len(qpsk_wave)}')
    
    
    #test time correction
    #delay_sec = (1/fs) * 10 # : 10 sample delay
    #new_t, new_signal = fractional_delay(t, qpsk_wave, delay_sec, fs)
    
    #test frequency correction
    new_signal = qpsk_wave * np.exp(-1j* 2 * np.pi * freq_offset * t) # shifts down by freq offset
    
    
    #tune to "close to baseband"
    tuned_sig = new_signal * np.exp(-1j * 2 * np.pi * sig_gen.freq * t)

    signal_ready = runCorrection(tuned_sig, fs, symb_rate)
    print(f'symbol length after runCorrection : {len(signal_ready)}') 
    #convert bits to string
    bits = np.zeros((len(signal_ready), 2), dtype=int)
    for i in range(len(signal_ready)):
        angle = np.angle(signal_ready[i], deg=True) % 360

        # codex for the phases to bits
        if 0 <= angle < 90:
            bits[i] = [0, 0]  # 45°
        elif 90 <= angle < 180:
            bits[i] = [0, 1]  # 135°
        elif 180 <= angle < 270:
            bits[i] = [1, 1]  # 225°
        else:
            bits[i] = [1, 0]  # 315°
    #concatennate these lists of lists
    bits = bits.flatten().tolist()
    
    print(f' the bits are {bits}')

    #bits to message:
    bits = bits[32:-32]
    bits = ''.join(str(bit) for bit in bits)
    # convert the bits into a string
    decoded_string = ''.join(chr(int(bits[i*8:i*8+8],2)) for i in range(len(bits)//8))
    print(f"The decoded message = {decoded_string}")

    # plt.plot(signal_ready)

    """
    plt.subplot(2, 1, 1)
    plt.plot(np.real(time_corrected), label = "Real")
    plt.subplot(2,1,2)
    plt.plot(np.imag(time_corrected), label = "Imag")
    plt.figure(figsize=(10, 5))
    """

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
    """
    
    # plt.subplot(2,2,3)
    # plt.title("Constellation: time_corrected")
    # plt.scatter(np.real(time_corrected), np.imag(time_corrected), s=2, alpha=0.5)
    # idx = np.arange(0, len(time_corrected), 1000)
    # plt.scatter(np.real(time_corrected)[idx], np.imag(time_corrected)[idx], color='red', s=20, label='Every 1000th')
    # plt.xlabel("In-phase")
    # plt.ylabel("Quadrature")
    # plt.legend()
    # plt.show()
    # qpsk_wave *= np.exp(-1j * 2 * np.pi * 900e6 * t) #baseband
    # offset_qpsk = frequency_offset(qpsk_wave, t)
    # coarse_fixed = coarse_freq_recovery(offset_qpsk)
    # costas_loop(coarse_fixed)

if __name__ == "__main__":
    main()
