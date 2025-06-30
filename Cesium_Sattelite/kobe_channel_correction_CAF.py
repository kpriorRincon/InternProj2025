import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from Sig_Gen import SigGen, rrc_filter
from scipy import signal
from config import *
freq_offset = 20000
fs = SAMPLE_RATE
symb_rate = SYMB_RATE
def integer_delay(num_samples, signal):
    '''Apply an integer delay by padding the front of the signal with zeros'''
    signal = np.concatenate([np.zeros(num_samples, dtype=complex), signal])
    t = np.arange(len(signal))/fs 
    return signal, t 

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
    N = NUMTAPS
    #construct filter
    n = np.linspace(-N//2, N//2,N)
    h = np.sinc(n-delay_in_samples)
    h *= np.hamming(N) #something like a rectangular window
    h /= np.sum(h) #normalize to get unity gain, we don't want to change the amplitude/power


    #apply filter: same time-aligned output with the same size as the input
    new_signal = np.convolve(signal, h, mode='full')
    delay = (N - 1) // 2
    new_signal = new_signal[delay:delay+len(signal)]

    if len(new_signal) == len(signal):
        print('signal lengths are the same: with mode = full and trimming')

    #create a new time vector with the correcct size
    new_t = np.arange(len(new_signal)) / Fs #longer time vector because the signal is delayed and padded in the front 
    
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
    alpha = 0.001 # immediate phase correction based on current error
    beta = 5e-7#  tracks accumalated phase error
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
    plt.figure()
    plt.plot(freq_log,'.-')
    plt.title('freq converge')
    plt.show()
    t = np.arange(len(qpsk_wave)) / fs
    return qpsk_wave * np.exp(-1j * 2 * np.pi * freq_log[-1] * t)
    
def CAF(incoming_signal,FS,symb_rate):
    import time 
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.signal import fftconvolve,resample_poly
    start_time = time.time()
    
    #create a list of frequencies 
    freqs = np.arange(-200, 201, 2)#create a frequency list 1-100
    #print(f'the frequency array: {freqs}') 
    '''
    correlations = [
    [...] correlation at -200 Hz
    [...] correlation at -199 Hz
    .
    .
    [...] correlation at 199 Hz
    [...] correlation at 200 Hz
    ]
    '''
    correlations = [] #create a correlation array that will store different correlations
    max_correlations = [] #store the max value and index max_val 
    
    #try interpolation before the CAF
    interpolated_incoming_signal = resample_poly(incoming_signal, 16, 1)     
    
    sig_gen = SigGen(0, 1.0) # we create a signal with a certain frequency 
    start_sequence = START_MARKER
    _, start_waveform = sig_gen.generate_qpsk(start_sequence)
    #be sure to interpolate the start marker as well 
    interpolated_start_waveform = resample_poly(start_waveform, 16, 1)
    t = np.arange(len(interpolated_start_waveform)) / (FS * 16)
    for freq in freqs:
        template = interpolated_start_waveform * np.exp(1j * 2 * np.pi * t * freq)     
        correlation = np.abs(fftconvolve(interpolated_incoming_signal, np.conj(np.flip(template)), mode = 'same'))
        correlations.append(correlation) 
        max_val = max(correlation)
        max_correlations.append(max_val)
   
    #now we can interpret which had the highest energy correlation by parsing the _max_correlations

    delta = time.time() - start_time
    print(f'function runtime: {delta}')
    #print(f'max_correlations:{max_correlations}') 

    best_correlation_index = max_correlations.index(max(max_correlations))
    print(f'max_correlation_index: {best_correlation_index}') 
    best_frequency = freqs[best_correlation_index]
    print(f'The best correlation happens with f offset of: {best_frequency}Hz')
    #find the index of the max value of the frequency slice with best correlation  
    idx = np.argmax(correlations[best_frequency]) - int(32 * (FS * 16)/symb_rate)
    #This is the og samples delay (will be fractional)
    delay_og_samples = idx / 16 
    
    # create a plot of the CAF
    X, Y = np.meshgrid(np.arange(len(correlations[0])), freqs)
    Z = np.array(correlations)
    max_idx = np.unravel_index(np.argmax(Z), Z.shape) #(row, col) -> (freq_index, delay_index)
    #print(f'max_idx: {max_idx}')
    max_freq = freqs[max_idx[0]]
    max_delay = max_idx[1]
    plt.figure(figsize=(10,6))
    plt.pcolormesh(X,Y,Z, shading='auto', cmap='plasma')
    plt.xlabel('Delay (fractional samples)')
    plt.ylabel('Frequency offset (Hz)')
    plt.colorbar(label='Correlation magnitude')
    plt.scatter(max_delay, max_freq, color='red', marker='s', s=50)
    plt.annotate('Peak Correlation', xy = (max_delay, max_freq), 
    xytext = (max_delay+20, max_freq+10), color = 'white', fontsize = 12)
    plt.title('2D Heatmap of Correlation')
    plt.show() 

    # Plot each correlation vs delay for each frequency as a colored line
    # color_list = ['r','g','b', 'm', 'c', 'k']
    # fig, ax = plt.subplots(figsize = (10, 6))
    # def animate(idx):
    #     ax.clear()
    #     freq = freqs[idx] 
    #     color = color_list[idx % len(color_list)]
    #     ax.plot(np.arange(len(correlations[idx])), correlations[idx], label = f'{freq} Hz', color = 'k')
    #     ax.set_xlabel('Delay (samples)')
    #     ax.set_ylabel('Correlation magnitude')
    #     ax.set_title(f'Correlation vs Dely\nFreq: {freq} Hz')
    #     ax.legend()
    # ani = animation.FuncAnimation(fig, animate, frames=len(freqs), interval = 300)
    # ani.save('correlation_animation.gif', writer='pillow', fps =len(freqs)//4)
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(10, 6))
    # for idx, freq in enumerate(freqs):
    #     color_for_this_iteration = color_list[idx % 6]
    #     plt.plot(np.arange(len(correlations[idx])), correlations[idx], label=f'{freq} Hz', color = color_for_this_iteration, linestyle = ':')
    # plt.xlabel('Delay (samples)')
    # plt.ylabel('Correlation magnitude')
    # plt.title('Correlation vs Delay for Each Frequency')
    # # Optionally, show legend for a subset of frequencies to avoid clutter
    # if len(freqs) <= 20:
    #     plt.legend()
    # else:
    #     step = max(1, len(freqs)//10)
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     plt.legend([handles[i] for i in range(0, len(handles), step)],
    #                [labels[i] for i in range(0, len(labels), step)],
    #                title='Frequency (Hz)')
    # plt.show()
    
    #fractionally delay the incoming singal with the amount 
    print(f'fractional delay accounted for: {delay_og_samples}') 
    t, signal = fractional_delay(np.arange(len(incoming_signal))/fs, incoming_signal, -delay_og_samples, fs) 
    #correct the signal in frequency
    
    shifted_sig = signal * np.exp(-1j*2*np.pi*best_frequency*t)
    
    #now find start and end with our known markers
    end_sequence = END_MARKER
    _, end_waveform = sig_gen.generate_qpsk(end_sequence)

    #find start index by convolving signal with preamble
    start_corr_sig = np.convolve(shifted_sig, np.conj(np.flip(start_waveform)), mode = 'same')
    # plt.figure()
    # plt.title('start correlation')
    # plt.plot(np.abs(start_corr_sig))
    
    #find the end index 
    end_corr_signal = np.convolve(shifted_sig, np.conj(np.flip(end_waveform)), mode = 'same')
    # plt.figure()
    # plt.plot(np.abs(end_corr_signal))
    # plt.title('end correlation')
   
    #get the index
    start = np.argmax(np.abs(start_corr_sig)) - int((32) * (FS / symb_rate)) # If the preamble is 32 bits long, its 16 symbols, symbols * samples/symbol = samples
    end = np.argmax(np.abs(end_corr_signal)) + int((32) * (FS / symb_rate))
    print(f'start: {start}, end: {end}')

    sig_ready = shifted_sig[start:end] 
    #since rx = htx, h = rx/tx so we can use our start sequence to correct for phase offset
    found_h = sig_ready[0 : int( 64 * (fs / symb_rate))] / start_waveform
    #normalize so theres no amplitude change and take the mean so we get avg phase offset of samples
    found_h_norm = np.mean(found_h / np.abs(sig_ready[0 : int( 64 * (fs / symb_rate))] / start_waveform))
    #debug:
    print(f'found h = {found_h_norm}')
    print(f'angle of h found = {np.rad2deg(np.angle(found_h_norm))}')

    sig_ready /= found_h_norm
    
    return sig_ready

def runCorrection(signal, FS, symbol_rate):
    from scipy.signal import fftconvolve
    
    #1. Coarse Freq Recovery 
    signal = coarse_freq_recovery(signal)    
   
    #2.Run CAF (matching with RRC version of signal achieves fine time correction)
    signal = CAF(signal, FS, symbol_rate) 
    
    #3. Fine Frequency Correction
    signal = costas_loop(signal, FS/symbol_rate)
    
    #4. Apply RRC to incoming signal turning the signal into -> raised cosine has the property small ISI
    #applying afterwards because I think the RRC disrupts the function of the fine frequency correction
    og_len = len(signal)
    _, h = rrc_filter(0.4, 301, 1/symbol_rate, FS)
    signal = fftconvolve(signal, h, mode = 'full')    
    delay = (301 - 1) // 2 #account for group delay
    signal = signal[delay:delay + og_len]
    
    return signal[::int(FS/symbol_rate)]

def main():
    sig_gen = SigGen(freq=900e6, amp=1)
    bits = sig_gen.message_to_bits('hello there ' * 3)
    print(f'num symbols: {len(bits)//2}')
    t, signal = sig_gen.generate_qpsk(bits)
    print(f'length of wave: {len(signal)}')
    
    #signal, t = integer_delay(100, signal)
    
    #test time correction
    delay_sec = 0.00232 # some random delay in seconds the fractional delay should be 0.6
    t, signal = fractional_delay(t, signal, delay_sec, fs)
    
    #test frequency correction
    new_signal = signal * np.exp(-1j* 2 * np.pi * freq_offset * t) # shifts down by freq offset
    THETA = np.random.uniform(-np.pi, np.pi)
    print(f'phase offset: {np.rad2deg(THETA)}')
    new_signal = new_signal * np.exp(1j * THETA)
    
    #tune to "close to baseband"
    tuned_sig = new_signal * np.exp(-1j * 2 * np.pi * sig_gen.freq * t)
    # Add AWGN noise to the tuned signal
    snr_db = 10  # Signal-to-noise ratio in dB
    signal_power = np.mean(np.abs(tuned_sig)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*tuned_sig.shape) + 1j * np.random.randn(*tuned_sig.shape))
    tuned_sig = tuned_sig + noise

    signal_ready = runCorrection(tuned_sig, fs, symb_rate)
    plt.figure()
    #outlier point 
    plt.plot(np.real(signal_ready[1:]),np.imag(signal_ready[1:]), 'o')
    plt.show() 
    print(f'Symbol length after runCorrection : {len(signal_ready)}') 
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
    bits = bits[128:-128]
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
