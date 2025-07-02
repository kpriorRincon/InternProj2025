import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import resample_poly, firwin, lfilter, fftconvolve
from Sig_Gen import SigGen, rrc_filter
from config import *
DEBUG = 1
freq_offset = 2000
time_delay = 0.00232
max_freq = 200
min_freq = -200
snr_db = 20

def integer_delay(signal, num_samples):
    '''Apply an integer delay by padding the front of the signal with zeros'''
    signal = np.concatenate([np.zeros(num_samples, dtype=complex), signal])
    t = np.arange(len(signal))/SAMPLE_RATE 
    return signal, t 

def fractional_delay(t, signal, delay):

    total_delay = delay * SAMPLE_RATE # seconds * samples/second = samples
    fractional_delay = total_delay % 1 # to get the remainder
    integer_delay = int(total_delay)
    print(f"Integer delay in samples: {integer_delay}")
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
    new_signal = fftconvolve(signal, h, mode='full')
    delay = (N - 1) // 2
    new_signal = new_signal[delay:delay+len(signal)]
    new_t = np.arange(len(new_signal)) / SAMPLE_RATE 

    return new_t, new_signal

def phase_offset(signal):
    theta = np.random.uniform(-np.pi, np.pi)
    new_signal = signal * np.exp(1j * theta)
    print(f'Random Phase Offset: {np.rad2deg(theta)}')

    return new_signal

def add_awgn(signal):
    sig_power = np.mean(np.abs(signal)**2)
    snr_lin = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_lin
    noise = np.sqrt(noise_power / 2) *(np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))

    return signal + noise

def lowpass_filter(raw_signal):
        lowcut = 0
        highcut = 0.05 #960e6
        fir_coeff = firwin(NUMTAPS, highcut, pass_zero=False, fs=SAMPLE_RATE)
        # pass-zero = whether DC / 0Hz is in the passband
        
        print(f"Length of raw signal: {len(raw_signal)}")
        filtered_sig = lfilter(fir_coeff, 1.0, raw_signal)
        print(f"Length of filtered signal: {len(filtered_sig)}")

        delay = (NUMTAPS - 1) // 2 
        padded_signal = np.pad(filtered_sig, (0, delay), mode='constant')
        filtered_sig = padded_signal[delay:] 


        return filtered_sig

def RRC_filter(signal):
    _, h = rrc_filter(BETA, NUMTAPS, 1/SYMB_RATE, SAMPLE_RATE)
    delay = (NUMTAPS - 1) // 2 
    rrc_signal = fftconvolve(signal, h, mode = 'full')    
    rrc_signal = rrc_signal[delay: delay + len(signal)]

    return rrc_signal

def decimate(signal, step):
    return signal[::int(step)]

def coarse_freq_recovery(qpsk_wave, order=4):


    qpsk_wave_r = qpsk_wave**4

    fft_vals = np.fft.fftshift(np.abs(np.fft.fft(qpsk_wave_r)))
    freqs = np.linspace(-SAMPLE_RATE/2, SAMPLE_RATE/2, len(fft_vals))

    freq_tone = freqs[np.argmax(fft_vals)] / order 
    print(f'frequency offset(coarse freq): {freq_tone}')
    
    t = np.arange(len(qpsk_wave)) / SAMPLE_RATE
    fixed_qpsk = qpsk_wave * np.exp(-1j*2*np.pi*freq_tone*t)

    if DEBUG:
        plt.plot(np.real(fixed_qpsk[1:]), np.imag(fixed_qpsk[1:]), 'b-', zorder = 1, label = 'oversampled signal')
        plt.scatter(np.real(fixed_qpsk[1::int(SAMPLE_RATE/SAMPLE_RATE)]),np.imag(fixed_qpsk[1::int(SAMPLE_RATE/SAMPLE_RATE)]), s=10, color= 'red', zorder = 2, label = 'decimated signal')
        plt.legend()
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.title('Coarse Frequency Synchronization')
        plt.savefig('media/coarse_correction.png')
        plt.close()

        plt.scatter(np.real(qpsk_wave[::int(SAMPLE_RATE/SYMB_RATE)]),np.imag(qpsk_wave[::int(SAMPLE_RATE/SYMB_RATE)]), s=10, color= 'blue', zorder = 2, label = 'RX Signal')
        plt.scatter(np.real(qpsk_wave[::int(SAMPLE_RATE/SYMB_RATE)]),np.imag(qpsk_wave[::int(SAMPLE_RATE/SYMB_RATE)]**2), s=10, color= 'green', zorder = 2, label = 'Signal at 2nd Power')
        plt.scatter(np.real(qpsk_wave_r[::int(SAMPLE_RATE/SYMB_RATE)]),np.imag(qpsk_wave_r[::int(SAMPLE_RATE/SYMB_RATE)]), s=10, color= 'red', zorder = 2, label = 'Signal at 4th Power')

        plt.legend()
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.title('Raising Signals to Nth Power')
        plt.axis()
        plt.grid()
        plt.show()
        plt.close()

        plt.plot(freqs, fft_vals)
        #plt.axvline(x=freq_tone * order, color='red', linestyle='--', label='Detected Tone')
        plt.legend()
        plt.xlim(-10e3, 100e3)
        plt.annotate(
            f'Frequency offset:\n{freq_tone * order:.2f} Hz',  # multi-line label in MHz
            xy=(freq_tone * order, np.max(fft_vals)),         # annotation target (converted to MHz)
            xytext=(freq_tone * order * 0.90, np.max(fft_vals) * 1.05 - 3),  # text to the left and slightly above
            fontsize=10,
            color='blue',
            ha='right',
            va='bottom',
            arrowprops=dict(facecolor='blue', arrowstyle='->', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='blue', alpha=0.8)
        )
        plt.xlabel('Frequency (kHz)')
        plt.grid()
        plt.ylabel('Magnitude')
        plt.title('FFT of Signal Raised to 4th')
        # label at freq_tone

        plt.show()



        
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
    freq = 0 # derivative of phase; rate of change of phase (radians/sample)
    #Following params determine feedback loop speed
    alpha = 0.001#0.027 #0.132 immediate phase correction based on current error
    beta = 5e-7#0.00286 #0.00932  tracks accumalated phase error
    out = np.zeros(N, dtype=np.complex64)
    freq_log = []
    
    for i in range(N):
        out[i] = qpsk_wave[i] * np.exp(-1j*phase) #adjust input sample by inv of estimated phase offset
        error = phase_detector_4(out[i])

        freq += (beta * error)
        #log frequency in Hz
        freq_log.append(freq * SAMPLE_RATE / (2 * np.pi))
        phase += freq + (alpha * error)

        while phase >= 2*np.pi:
            phase -= 2*np.pi
        while phase < 0:
            phase += 2*np.pi
    
    #finds the frequency at the end when it converged
    print(f'Costas Converged Frequency Offset: {freq_log[-1]}')

    t = np.arange(len(qpsk_wave)) / SAMPLE_RATE
    fixed_qpsk = qpsk_wave * np.exp(-1j * 2 * np.pi * freq_log[-1] * t)

    if DEBUG:
        plt.figure(figsize=(10, 6))
        plt.plot(freq_log,'.-')
        plt.title('Costas Loop Frequency Convergence')
        plt.grid()
        plt.xlabel('Loop Iteration Count')
        plt.ylabel('Frequency (Hz)')
        plt.savefig('media/costas_convergence.png')
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.plot(np.real(fixed_qpsk[1:-1]),np.imag(fixed_qpsk[1:-1]), 'b-', label='oversampled signal', zorder = 1)
        plt.scatter(np.real(fixed_qpsk[1:-1:int(SAMPLE_RATE/SYMB_RATE)]),np.imag(fixed_qpsk[1:-1:int(SAMPLE_RATE/SYMB_RATE)]), s=10, color= 'red', zorder = 2, label = 'decimated signal')
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.grid(True)
        plt.title('Fine Frequency Synchronization (Costas Loop)')
        plt.savefig('media/fine_correction.png')
        plt.close()

    return fixed_qpsk
    

def mixing(signal, f, ip_val=1):
    t = np.arange(len(signal)) / (SAMPLE_RATE * ip_val)
    return signal * np.exp(1j * 2 * np.pi * f * t)

def correlate(signal, match_filter):
    # Convolve filter with signal and extract highest correlation
    energy_map = fftconvolve(signal, np.conj(np.flip(match_filter)), mode = 'same')
    highest_correlation = np.max(np.abs(energy_map))    

    return highest_correlation

visited_freqs = []
correlation_values = []
def binary_search(rx_signal, match_filter, l_freq, r_freq):
    # Base case: 
    if l_freq >= r_freq:
        return l_freq
    
    # get middle index of each segment when test frequency divided into 2
    l_match_filter = mixing(match_filter, l_freq, INTERPOLATION_VAL)
    r_match_filter = mixing(match_filter, r_freq, INTERPOLATION_VAL)

    l_energy = correlate(rx_signal, l_match_filter)
    r_energy = correlate(rx_signal, r_match_filter)

    visited_freqs.extend([l_freq, r_freq])
    correlation_values.extend([l_energy, r_energy])

    print(f"Energy at {l_freq} Hz: {l_energy}. Energy at {r_freq} Hz: {r_energy}.")
    if r_energy > l_energy:
        l_freq += (r_freq - l_freq) // 2 + 1
        freq = binary_search(rx_signal, match_filter, l_freq, r_freq)
    
    elif r_energy < l_energy:
        r_freq -= (r_freq - l_freq) // 2 + 1
        freq = binary_search(rx_signal, match_filter, l_freq, r_freq)

    else:
        # Offset exactly in between, choose middle freq
        freq = l_freq + (r_freq - l_freq) // 2
    
    return freq


def cross_corr_caf(rx_signal):
    # Generate QPSK wave of start marker
    sig_gen = SigGen(0, 1.0)    
    _, marker_filter = sig_gen.generate_qpsk(START_MARKER)

    #Interpolate signal and match filter
    ip_signal = resample_poly(rx_signal, INTERPOLATION_VAL, 1)
    ip_filter = resample_poly(marker_filter, INTERPOLATION_VAL, 1)

    strt = time.time()
    # Binary search for frequency offset
    freq_found = binary_search(ip_signal, ip_filter, min_freq, max_freq)

    print(f"Total time for binary search: {time.time() - strt} s.")
    print(f"Binary Search CAF: {freq_found}")

    if DEBUG:
        # Binary search CAF convergence plot
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(visited_freqs)), visited_freqs, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Frequency Offset (Hz or bins)")
        plt.title("Binary Search Convergence on Frequency Offset")
        plt.grid(True)
        plt.savefig('media/binary_search_convergence.png')
        plt.close()

    #Correlate one last time to get index
    up_mixed_filter = mixing(ip_filter, freq_found, INTERPOLATION_VAL)
    start_map = fftconvolve(ip_signal, np.conj(np.flip(up_mixed_filter)), mode = 'same')
    start_idx = np.argmax(np.abs(start_map)) - int((32) * (SAMPLE_RATE * INTERPOLATION_VAL / SYMB_RATE))

    if DEBUG:
        # Start marker correlation graph
        plt.figure(figsize = (10, 6))
        plt.title('Start Correlation')
        plt.plot(np.abs(start_map))
        plt.xlabel(f'Fractional Sample Index (interpolation rate: {INTERPOLATION_VAL})')
        plt.ylabel('Correlation Magnitude')
        plt.savefig('media/start_correlation.png')
        plt.close()

    #Correlate with end marker match filter for end idx
    _, end_filter = sig_gen.generate_qpsk(END_MARKER)
    ip_end_filter = resample_poly(end_filter, INTERPOLATION_VAL, 1)
    mixed_end_filter = mixing(ip_end_filter, freq_found, INTERPOLATION_VAL)
    end_map = fftconvolve(ip_signal, np.conj(np.flip(mixed_end_filter)), mode = 'same')
    end_idx = np.argmax(np.abs(end_map)) + int((32) * (SAMPLE_RATE * INTERPOLATION_VAL / SYMB_RATE))

    if DEBUG:
        # End marker correlation graph
        plt.figure(figsize=(10, 6))
        plt.title('End Correlation')
        plt.xlabel('Sample Index')
        plt.ylabel('Correlation Magnitude')
        plt.plot(np.abs(end_map))
        plt.xlabel(f'Fractional Sample Index (interpolation rate: {INTERPOLATION_VAL})')
        plt.ylabel('Correlation Magnitude')
        plt.savefig('media/end_correlation.png')
        plt.close()

        # Plot both correlations on top of the signal 
        plt.figure(figsize=(10, 6))
        plt.plot(np.abs(start_map), label='Start Marker Correlation', alpha=0.7)
        plt.plot(np.abs(end_map), label='End Marker Correlation', alpha=0.7)
        plt.axvline(start_idx, color='g', linestyle='--', label='Start Index')
        plt.axvline(end_idx, color='r', linestyle='--', label='End Index')
        plt.xlim(start_idx - 20000, end_idx + 2000)  # Adjust x-axis limits for better visibility
        plt.title('Start and End Marker Correlation')
        plt.xlabel(f'Fractional Sample Index (interpolation rate: {INTERPOLATION_VAL})')
        plt.ylabel('Correlation Magnitude')
        plt.legend()
        plt.grid(True)
        plt.savefig('media/start_end_correlation.png')
        plt.close()

    # Reslice signal
    print(f"Start: {start_idx} End: {end_idx}")
    deci_signal = ip_signal[start_idx: end_idx:16]   
    if DEBUG:
        plt.figure(figsize=(6, 6))
        plt.plot(np.real(deci_signal[1:]), np.imag(deci_signal[1:]), 'b-', zorder = 1, label = 'oversampled signal')
        plt.scatter(np.real(deci_signal[1::int(SAMPLE_RATE/SYMB_RATE)]), np.imag(deci_signal[1::int(SAMPLE_RATE/SYMB_RATE)]), s=10, color= 'red', zorder = 2, label = 'decimated signal')
        plt.legend()
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.title('Output of CAF')
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('media/pre_phase_correction_constellation.png')
        plt.close()



    # Fix phase offset
    sig_start = deci_signal[0: int(64 * SAMPLE_RATE / SYMB_RATE)]
    h =  sig_start / marker_filter
    h_norm = np.mean(h / np.abs(h))
    print(f'Phase offset found: {np.rad2deg(np.angle(h_norm))}')

    deci_signal /= h_norm

    t = np.arange(len(deci_signal)) / SAMPLE_RATE
    fixed_signal = deci_signal * np.exp(-1j * 2 * np.pi * freq_found * t)

    if DEBUG:
        # Plotting the unit circle
        # Calculate phase in degrees and radians
        phase_rad = np.angle(h_norm)
        phase_deg = np.rad2deg(phase_rad)
        print(f'Phase offset found: {phase_deg:.2f} degrees')
        phase = np.angle(h_norm)
        plt.figure(figsize=(6, 6))
        # Plot the unit circle
        circle = plt.Circle((0, 0), 1, color='lightgray', fill=False, linestyle='--')
        plt.gca().add_artist(circle)

        # Plot the arc from 0 to phase
        arc_theta = np.linspace(0, phase, 100)
        plt.plot(np.cos(arc_theta), np.sin(arc_theta), color='orange', linewidth=2, label='Phase Arc')

        # Plot the vector for h
        plt.plot([0, np.real(h_norm)], [0, np.imag(h_norm)], marker='o', color='b', label='h normalized')

        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.title(f'Detected Phase offset: {np.degrees(phase):.2f}°')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.tight_layout()
        plt.plot(np.real(fixed_signal[1:]),np.imag(fixed_signal[1:]), 'o')
        plt.savefig('media/phase_offset.png', dpi = 300)
        plt.close()

        # Plot the constellation after phase correction
        plt.figure(figsize=(6, 6))
        plt.plot(np.real(fixed_signal[1:]), np.imag(fixed_signal[1:]), 'b-', zorder = 1, label = 'oversampled signal')
        plt.scatter(np.real(fixed_signal[1::int(SAMPLE_RATE/SYMB_RATE)]), np.imag(fixed_signal[1::int(SAMPLE_RATE/SYMB_RATE)]), s=10, color= 'red', zorder = 2, label = 'decimated signal')
        plt.legend()
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.title('Constellation after Phase Correction')
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('media/phase_corrected_constellation.png')
        plt.close()
    return fixed_signal

def demodulator(qpsk_sig):
    #convert bits to string
    bits = np.zeros((len(qpsk_sig), 2), dtype=int)
    for i in range(len(qpsk_sig)):
        angle = np.angle(qpsk_sig[i], deg=True) % 360

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
    
    bits = bits[128:-128]
    bits = ''.join(str(bit) for bit in bits)
    # convert the bits into a string
    
    decoded_string = ''.join(chr(int(bits[i*8:i*8+8],2)) for i in range(len(bits)//8))
    return decoded_string

def channel_handler(rx_signal):
    if DEBUG:
        plt.figure(figsize=(6, 6))
        plt.plot(np.real(rx_signal[1:]), np.imag(rx_signal[1:]), 'b-', zorder = 1, label = 'oversampled signal')
        plt.scatter(np.real(rx_signal[1::int(SAMPLE_RATE/SYMB_RATE)]), np.imag(rx_signal[1::int(SAMPLE_RATE/SYMB_RATE)]), c='r',s = 30, zorder = 2, label = 'decimated signal')
        plt.legend()
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.grid(True)
        plt.axis('equal')
        plt.title('Incoming IQ Plot')
        plt.savefig('media/rx_incoming.png')
        plt.close()

    filtered_sig = lowpass_filter(rx_signal)
    coarse_fixed = coarse_freq_recovery(filtered_sig)
    caf_fixed = cross_corr_caf(coarse_fixed)
    costas_fixed = costas_loop(caf_fixed)
    rrc_signal = RRC_filter(costas_fixed)
    signal_ready = decimate(rrc_signal, int(SAMPLE_RATE/SYMB_RATE))
    decoded_message = demodulator(signal_ready)
    if DEBUG:
            plt.figure(figsize=(6, 6))
            plt.plot(np.real(signal_ready[64:-64]), np.imag(signal_ready[64:-64]), 'o')
            plt.title('Final IQ Plot')
            plt.grid(True)
            plt.xlabel('In-Phase (I)')
            plt.ylabel('Quadrature (Q)')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig('media/clean_signal.png')
            plt.close()
    return decoded_message

def main():
    #Generate QPSK at Carrier Frequency
    sig_gen = SigGen(freq=900e6, amp=1)
    bits = sig_gen.message_to_bits('hello there ' * 3)
    t, qpsk_wave = sig_gen.generate_qpsk(bits)   

    print(f"Length of TX Signal: {len(qpsk_wave)}")
    # Integer time delay
    #qpsk_wave, t = integer_delay(qpsk_wave, 100)
    #t, qpsk_wave = fractional_delay(t, qpsk_wave, time_delay)

    # Set frequency offset
    #qpsk_wave = qpsk_wave * np.exp(1j* 2 * np.pi * freq_offset * t) # shifts down by freq offset
    
    # Uniformly random phase offset
    #qpsk_wave = phase_offset(qpsk_wave)

    # Adding AWGN
    post_channel_wave = add_awgn(qpsk_wave)

    #Tune down to baseband
    qpsk_base = post_channel_wave * np.exp(-1j * 2 * np.pi * sig_gen.freq * t)

    lpf_signal = lowpass_filter(qpsk_base)
    
    coarse_fixed_sig = coarse_freq_recovery(lpf_signal)

    # Run CAF and return frequency offset found with highest correlation
    caf_fixed_sig = cross_corr_caf(coarse_fixed_sig)

    #Down convert with offset
    final_fixed_sig = costas_loop(caf_fixed_sig)

    # Pass through RRC filter
    rc_signal = RRC_filter(final_fixed_sig)

    # Decimate
    signal_ready = decimate(rc_signal, int(SAMPLE_RATE/SYMB_RATE))
    # Demodulate qpsk and display message
    message = demodulator(signal_ready)
    
    if DEBUG:
        plt.figure(figsize=(6, 6))
        plt.plot(np.real(signal_ready[64:-64]), np.imag(signal_ready[64:-64]), 'o')
        plt.title('Final IQ Plot')
        plt.grid(True)
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('media/clean_signal.png')
        plt.close()

    #message = channel_handler(qpsk_base)
    print(f"The decoded message = {message}")

if __name__ == "__main__":
    main()