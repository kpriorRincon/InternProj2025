import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import resample_poly, firwin, lfilter, fftconvolve
from Sig_Gen import SigGen, rrc_filter
from config import *
DEBUG = 1
freq_offset = 20000
time_delay = 0.00232
max_freq = 200
min_freq = -200
snr_db = 10

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
    _, h = rrc_filter(0.4, NUMTAPS, 1/SYMB_RATE, SAMPLE_RATE)
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
        plt.plot(np.real(fixed_qpsk[1:]),np.imag(fixed_qpsk[1:]), 'o')
        plt.savefig('media/coarse_correction.png')
        plt.close()
        
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

    if DEBUG:
        plt.plot(freq_log,'.-')
        plt.title('freq converge')
        plt.clf()
    t = np.arange(len(qpsk_wave)) / SAMPLE_RATE
    return qpsk_wave * np.exp(-1j * 2 * np.pi * freq_log[-1] * t)
    

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
        plt.plot(range(len(visited_freqs)), visited_freqs, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Frequency Offset (Hz or bins)")
        plt.title("Binary Search Convergence on Frequency Offset")
        plt.grid(True)
        plt.savefig('media/binary_search_convergence.png')
        plt.clf()

    #Correlate one last time to get index
    up_mixed_filter = mixing(ip_filter, freq_found, INTERPOLATION_VAL)
    start_map = fftconvolve(ip_signal, np.conj(np.flip(up_mixed_filter)), mode = 'same')
    start_idx = np.argmax(np.abs(start_map)) - int((32) * (SAMPLE_RATE * INTERPOLATION_VAL / SYMB_RATE))

    if DEBUG:
        # Start marker correlation graph
        plt.title('start correlation')
        plt.plot(np.abs(start_map))
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
        plt.title('ends correlation')
        plt.plot(np.abs(end_map))
        plt.savefig('media/end_correlation.png')
        plt.close()

    # Reslice signal
    print(f"Start: {start_idx} End: {end_idx}")
    deci_signal = ip_signal[start_idx: end_idx:16]   

    # Fix phase offset
    sig_start = deci_signal[0: int(64 * SAMPLE_RATE / SYMB_RATE)]
    h =  sig_start / marker_filter
    h_norm = np.mean(h / np.abs(h))
    print(f'Phase offset found: {np.rad2deg(np.angle(h_norm))}')

    deci_signal /= h_norm

    if DEBUG:
        # Plotting the unit circle
        # Calculate phase in degrees and radians
        phase_rad = np.angle(h_norm)
        phase_deg = np.rad2deg(phase_rad)
        print(f'Phase offset found: {phase_deg:.2f} degrees')

        point = h_norm / np.abs(h_norm)

        # Create unit circle plot
        fig, ax = plt.subplots(figsize=(6,6))
        circle = plt.Circle((0, 0), 1, color='lightgray', fill=False, linestyle='--')
        ax.add_artist(circle)

        # Plot the point
        ax.plot(point.real, point.imag, 'bo', label='h_norm')

        # Draw the curved red arc from angle 0 to phase_rad
        theta = np.linspace(0, phase_rad, 100)
        x_arc = np.cos(theta)
        y_arc = np.sin(theta)
        ax.plot(x_arc, y_arc, 'r-', linewidth=2, label='Phase arc')

        # Dashed red line from center to point
        ax.plot([0, point.real], [0, point.imag], 'r--', linewidth=1)

        # Plot x-axis line in light gray for reference
        ax.plot([0, 1], [0, 0], color='gray', linestyle='--')

        # Annotation box text
        annot_text = f'Phase: {phase_deg:.1f}°\nValue: {point.real:.2f} + {point.imag:.2f}j'

        # Annotate near the point with an arrow
        ax.annotate(
            annot_text,
            xy=(point.real, point.imag),
            xytext=(point.real + 0.1, point.imag + 0.1),  # offset text a bit
            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
        )

        # Setup plot limits and labels
        ax.set_xlim(-1.1, 1.3)
        ax.set_ylim(-1.1, 1.3)
        ax.set_aspect('equal')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add legend and title
        ax.set_title('Phase Offset Detected')

        plt.savefig('media/phase_offset.png')
        plt.close()
        
    t = np.arange(len(deci_signal)) / SAMPLE_RATE
    fixed_signal = deci_signal * np.exp(-1j * 2 * np.pi * freq_found * t)

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

    filtered_sig = lowpass_filter(rx_signal)
    coarse_fixed = coarse_freq_recovery(filtered_sig)
    caf_fixed = cross_corr_caf(coarse_fixed)
    costas_fixed = costas_loop(caf_fixed)
    rrc_signal = RRC_filter(costas_fixed)
    signal_ready = decimate(rrc_signal, int(SAMPLE_RATE/SYMB_RATE))
    decoded_message = demodulator(signal_ready)

    return decoded_message

def main():
    #Generate QPSK at Carrier Frequency
    sig_gen = SigGen(freq=900e6, amp=1)
    bits = sig_gen.message_to_bits('hello there' * 3)
    t, qpsk_wave = sig_gen.generate_qpsk(bits)    
    print(f"Length of TX Signal: {len(qpsk_wave)}")
    # Integer time delay
    #qpsk_wave, t = integer_delay(qpsk_wave, 100)
    t, qpsk_wave = fractional_delay(t, qpsk_wave, time_delay)

    # Set frequency offset
    qpsk_wave = qpsk_wave * np.exp(1j* 2 * np.pi * freq_offset * t) # shifts down by freq offset
    
    # Uniformly random phase offset
    qpsk_wave = phase_offset(qpsk_wave)

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
    
    #message = channel_handler(qpsk_base)
    print(f"The decoded message = {message}")

if __name__ == "__main__":
    main()