from Sig_Gen import SigGen, rrc_filter
import numpy as np



fs = 2.88e6
symb_rate = fs/20
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

def CAF(incoming_signal, FS, symb_rate):
    orig_len = len(incoming_signal)
    delay = (301 - 1) //2 #group delay of FIR filter is always (N-1)/2 samples N is num taps
    _, h = rrc_filter(0.4, 301, 1/symb_rate, FS)
    incoming_signal = np.convolve(incoming_signal, h, mode = 'full')
    incoming_signal = incoming_signal[delay : delay + orig_len]

    #create a list of frequencies 
    freqs = np.arange(-250, 251, .5)#create a frequency list 1 - 100
    '''
    correlations = [
    [...] correlation at -100 Hz
    [...] correlation at -99 Hz
    .
    .
    [...] correlation at 99 Hz
    [...] correlation at 100 Hz
    ]
    '''
    correlations = [] #create a correlation array that will store different correlations
    max_correlations = [] #store the max value and index max_val 
    
    for freq in freqs:
        
        #create a template at each frequency
        sig_gen = SigGen(freq, 1.0, FS, symb_rate) # we create a signal with a certain frequency 
        # 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 0 0 0 1 0 1 0 1 1 1 0 1 1 0 0 0 1
        start_sequence = sig_gen.start_sequence
        _, start_waveform = sig_gen.generate_qpsk(start_sequence)

        #then apply RRC to make a full RC filter to compare agianst
        og_len = len(start_waveform)
        start_waveform = np.convolve(start_waveform, h, mode = 'full')
        start_waveform = start_waveform[delay:delay + og_len]
        correlation = np.abs(np.convolve(incoming_signal, np.conj(np.flip(start_waveform)), mode = 'same'))
        #plt.figure()
        #plt.plot(correlation)
        #plt.show()  
        correlations.append(correlation) 
        max_val = max(correlation)
        max_correlations.append(max_val)
   
    #now we can interpret which had the highest energy correlation by parsing the _max_correlations
    #print(f'max_correlations:{max_correlations}') 
    best_correlation_index = max_correlations.index(max(max_correlations))
    best_frequency = freqs[best_correlation_index]

    #correct the signal
    t = np.arange(len(incoming_signal))/fs
    shifted_sig = incoming_signal*np.exp(-1j*2*np.pi*best_frequency*t)
    
    #now find start and end with our known markers
   
    sig_gen = SigGen(0, 1.0, FS, symb_rate) # if f = 0 this won't up mix so we'll get the baseband signal 
    # 1 1 1 1 1 0 0 1 1 0 1 0 0 1 0 0 0 0 1 0 1 0 1 1 1 0 1 1 0 0 0 1
    start_sequence = sig_gen.start_sequence

    _, start_waveform = sig_gen.generate_qpsk(start_sequence)
    start_orig_len = len(start_waveform)
    start_waveform = np.convolve(start_waveform, h, mode = 'full')    
    start_waveform = start_waveform[delay: delay + start_orig_len]

    # 0 0 1 0 0 1 1 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0 1 0 0 1 0
    end_sequence = sig_gen.end_sequence
    
    _, end_waveform = sig_gen.generate_qpsk(end_sequence)
    end_orig_len = len(end_waveform)
    end_waveform = np.convolve(end_waveform, h, mode = 'full')    
    end_waveform = end_waveform[delay: delay + end_orig_len]

    #find start index by convolving signal with preamble
    start_corr_sig = np.convolve(shifted_sig, np.conj(np.flip(start_waveform)), mode = 'same')

    #find the end index 
    end_corr_signal = np.convolve(shifted_sig, np.conj(np.flip(end_waveform)), mode = 'same')
   
    #get the index
    start = np.argmax(np.abs(start_corr_sig)) - int((32) * (FS/symb_rate)) # If the preamble is 32 bits long, its 16 symbols, symbols * samples/symbol = samples
    end = np.argmax(np.abs(end_corr_signal)) + int((8) * (FS/symb_rate))
    print(f'start: {start}, end: {end}')


    sig_ready = shifted_sig[start:end] 


    return sig_ready, best_frequency
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
    sig_gen = SigGen(freq = 900e6, amp = 1 ,sample_rate = fs, symbol_rate = symb_rate)
    bits = sig_gen.message_to_bits('hello there ' * 3)
    t, qpsk_wave = sig_gen.generate_qpsk(bits)

    # Set frequency Offset
    qpsk_wave = qpsk_wave * np.exp(1j* 2 * np.pi * freq_offset * t) # shifts up by freq offset
    
    #Tune down to baseband
    tuned_sig = qpsk_wave * np.exp(-1j * 2 * np.pi * sig_gen.freq * t)

    coarse_fixed_sig, coarse_freq = coarse_freq_recovery(tuned_sig)

    print(f"Coarse Frequency Correction: {coarse_freq} Hz")

    sig, frequency_CAF = CAF(coarse_fixed_sig, fs, symb_rate)
    print(f'CAF frequency Correction: {frequency_CAF} Hz')
    
    best_alpha = None
    best_beta = None
    min_error = float('inf')
    best_correction = None

    alphas = np.arange(0, 0.01, 0.00005)
    betas = np.arange(0, 0.00001, 0.00000005)

    print(f"Testing {len(alphas)} alphas and {len(betas)} betas.")
    print(f"Total number of test cases: {len(alphas) * len(betas)}.")

    counter = 0
    for a in alphas:
        for b in betas:
            fine_freq = costas_loop(sig, alpha=a, beta=b)
            total_correction = coarse_freq + frequency_CAF + fine_freq 
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
