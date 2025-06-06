import numpy as np
import matplotlib.pyplot as plt
def Plotter(sample_rate, t, tx_signal, tx_vert_lines, symbol_rate, tx_symbols, sig_gen_mapping, message, rep_incoming_signal,rep_mixed_signal, rep_filtered_signal, rx_analytical_signal):
    #this plot is for time qpsk
                    plt.figure(figsize=(15, 5))
                    plt.plot(t, tx_signal)
                    plt.ylim(-1/np.sqrt(2)*1-.5, 1/np.sqrt(2)*1+.5)

                    #if there are more than 10 symbols only show the first ten symbols
                    if len(tx_vert_lines) > 10:
                        plt.xlim(0, 10/symbol_rate)  # Show first 10 symbol periods
                    #if not don't touch the xlim

                    for lines in tx_vert_lines:
                        #add vertical lines at the symbol boundaries
                        if len(tx_vert_lines) > 10:
                            if lines < 10/symbol_rate:
                                plt.axvline(x=lines, color='black', linestyle='--', linewidth=1)

                                #add annotation for the symbol e.g. '00', '01', '10', '11'
                                # Reverse mapping: symbol -> binary pair
                                symbol = tx_symbols[tx_vert_lines.index(lines)]
                                # Reverse the mapping to get binary pair from symbol
                                reverse_mapping = {v: k for k, v in sig_gen_mapping.items()}
                                binary_pair = reverse_mapping.get(symbol, '')
                                formatted_pair =str(binary_pair).replace("(", "").replace(")", "").replace(", ", "")
                                #debug
                                #print(formatted_pair)
                                x_dist = 1 / (2.7 * symbol_rate) #half the symbol period 
                                y_dist = 0.707*1 + .2 # 0.807 is the amplitude of the QPSK waveform
                                plt.annotate(formatted_pair, xy=(lines, 0), xytext=(lines + x_dist, y_dist), fontsize=17)  
                        else:
                            if lines < len(t):
                                plt.axvline(x=lines, color='black', linestyle='--', linewidth=1)

                                #add annotation for the symbol e.g. '00', '01', '10', '11'
                                # Reverse mapping: symbol -> binary pair
                                symbol = tx_symbols[tx_vert_lines.index(lines)]
                                # Reverse the mapping to get binary pair from symbol
                                reverse_mapping = {v: k for k, v in sig_gen_mapping.items()}
                                binary_pair = reverse_mapping.get(symbol, '')
                                formatted_pair =str(binary_pair).replace("(", "").replace(")", "").replace(", ", "")
                                #debug
                                #print(formatted_pair)
                                x_dist = 1 / (2.7 * symbol_rate) #half the symbol period 
                                y_dist = 0.707*1 + .2 # 0.807 is the amplitude of the QPSK waveform
                                plt.annotate(formatted_pair, xy=(lines, 0), xytext=(lines + x_dist, y_dist), fontsize=17)
                            
                    if len(tx_vert_lines) > 10:
                        plt.title(f'QPSK Waveform for \"{message}\" (first 10 symbol periods)')
                    else:
                        plt.title(f'QPSK Waveform for \"{message}\"')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Amplitude')
                    plt.grid()
                    # Save the plot to a file
                    plt.savefig(f'qpsk_sig_gen/1_qpsk_waveform.png', dpi=300)    
                    #print("Debug: plot generated")
                    #END of time plot

                    #frequency for qpsk tx
                    #same figure size as above
                    n = len(t)
                    freqs = np.fft.fftfreq(n, d=1/sample_rate)
                    # FFT of original and shifted signals
                    fft = np.fft.fft(tx_signal)
                    fft_db = 20 * np.log10(np.abs(fft))
                    # get fft of qpsk signal
                    
                    plt.figure(figsize=(15, 5))
                    plt.plot(freqs,fft_db)
                    plt.title("FFT of QPSK signal")
                    plt.xlim(0, 1000e6)
                    plt.ylim(0, np.max(fft_db)+10)
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Magnitude (dB)')
                    #save plot
                    plt.savefig(f'qpsk_sig_gen/2_qpsk_waveform.png', dpi = 300)

                    #end plot for qpsk

                    #start plots for repeater: 
                    x_t_lim = 3 / symbol_rate
                    n = len(t)
                    freqs = np.fft.fftfreq(n, d=1/sample_rate)
                    positive_freqs = freqs > 0
                    positive_freq_values = freqs[positive_freqs]

                    # FFT of original and shifted signals
                    fft_input = np.fft.fft(rep_incoming_signal)
                    fft_shifted = np.fft.fft(rep_mixed_signal)
                    fft_filtered = np.fft.fft(rep_filtered_signal)
                    # Convert magnitude to dB
                    mag_input = 20 * np.log10(np.abs(fft_input))
                    mag_shifted = 20 * np.log10(np.abs(fft_shifted))
                    mag_filtered = 20 * np.log10(np.abs(fft_filtered))
                    plt.figure(figsize=(20, 6))

                    # --- Time-domain plot: Original QPSK ---
                    plt.subplot(1, 2, 1)
                    plt.plot(t, np.real(rep_incoming_signal))  # convert time to microseconds
                    plt.title("Original QPSK Signal (Time Domain)")
                    plt.xlabel("Time (μs)")
                    plt.ylabel("Amplitude")
                    plt.xlim(0, x_t_lim)
                    plt.grid(True)

                    plt.subplot(1, 2, 2)
                    positive_mags = mag_input[positive_freqs]
                    positive_freq_values = freqs[positive_freqs]
                    peak_index = np.argmax(positive_mags)
                    peak_freq = positive_freq_values[peak_index]
                    peak_mag = positive_mags[peak_index]
                    plt.plot(freqs, mag_input, label="Original QPSK", alpha=0.8)
                    plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq/1e6:.1f} MHz')
                    plt.text(peak_freq, peak_mag + 5, f'{peak_freq/1e6:.1f} MHz', color='r', ha='center')
                    #plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
                    plt.xlabel("Frequency (GHz)")
                    plt.ylabel("Magnitude (dB)")
                    plt.title("FFT of QPSK Before Frequency Shift")
                    plt.xlim(0, sample_rate / 2)  # From 0 to fs in MHz
                    plt.ylim(0, np.max(mag_input) + 10)
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('original_qpsk_rp.png')

                    plt.clf()

                    # --- Time-domain plot: Shifted QPSK ---
                    plt.subplot(1, 2, 1)
                    plt.plot(t, np.real(rep_mixed_signal))
                    plt.title("Shifted QPSK Signal (Time Domain)")
                    plt.xlabel("Time (μs)")
                    plt.ylabel("Amplitude")
                    plt.xlim(0, x_t_lim)
                    plt.grid(True)

                    

                    plt.subplot(1, 2, 2)
                    positive_mags = mag_shifted[positive_freqs]
                    positive_freq_values = freqs[positive_freqs]
                    peak_index = np.argmax(positive_mags)
                    peak_freq = positive_freq_values[peak_index]
                    peak_mag = positive_mags[peak_index]
                    plt.plot(freqs, mag_shifted, label="Shifted QPSK", alpha=0.8)
                    plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq/1e6:.1f} MHz')
                    plt.text(peak_freq, peak_mag + 5, f'{peak_freq/1e6:.1f} MHz', color='r', ha='center')
                    plt.xlabel("Frequency (GHz)")
                    plt.ylabel("Magnitude (dB)")
                    plt.title("FFT of QPSK After Frequency Shift")
                    plt.xlim(0, sample_rate / 2)  # From 0 to fs in MHz
                    plt.ylim(0, np.max(mag_input) + 10)
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()

                    plt.savefig('shifted_qpsk_rp.png')
                    plt.clf()

                    plt.subplot(1, 2, 1)
                    plt.plot(t, np.real(rep_filtered_signal))
                    plt.title("Filtered QPSK Signal (Time Domain)")
                    plt.xlabel("Time (μs)")
                    plt.ylabel("Amplitude")
                    plt.xlim(0, x_t_lim)
                    plt.grid(True)

                    plt.subplot(1, 2, 2)
                    positive_mags = mag_filtered[positive_freqs]
                    positive_freq_values = freqs[positive_freqs]
                    peak_index = np.argmax(positive_mags)
                    peak_freq = positive_freq_values[peak_index]
                    peak_mag = positive_mags[peak_index]
                    #print(freqs[peak_index-3:peak_index+3])
                    plt.plot(freqs, mag_filtered, label="Filtered QPSK", alpha=0.8)
                    plt.axvline(x=peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq/1e6:.1f} MHz')
                    plt.text(peak_freq, peak_mag + 5, f'{peak_freq/1e6:.1f} MHz', color='r', ha='center')
                    plt.xlabel("Frequency (GHz)")
                    plt.ylabel("Magnitude (dB)")
                    plt.title("FFT of QPSK After Filtering")
                    plt.xlim(0, sample_rate / 2)  # From 0 to fs in MHz
                    plt.ylim(0, np.max(mag_input) + 10)
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()

                    plt.savefig('filtered_qpsk_rp.png')
                    plt.clf()
                    #end repeater plotting


                    #start receiver plotting:
                    # constellation plot
                    plt.figure(figsize=(5, 5))

                    plt.plot(np.real(rx_analytical_signal), np.imag(rx_analytical_signal), '.')
                    plt.grid(True)
                    plt.title('Constellation Plot of Sampled Symbols')
                    plt.xlabel('Real')
                    plt.xlim(-1,1)
                    plt.ylim(-1,1)
                    plt.ylabel('Imaginary')
                    plt.tight_layout()
                    plt.savefig('demod_media/Constellation.png')


                    # plot the fft
                    ao_fft = np.fft.fft(rx_analytical_signal)
                    freqs = np.fft.fftfreq(len(rx_analytical_signal), d=1/2*sample_rate)
                    plt.figure(figsize=(10, 4))
                    plt.plot(freqs, 20*np.log10(ao_fft))
                    plt.title('FFT of the Base Band Signal')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Madgnitude (dB)')
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig('demod_media/Base_Band_FFT.png')