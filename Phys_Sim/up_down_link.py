# imports
import numpy as np
import matplotlib.pyplot as plt

#### predefined parameters ####
T = 290             # noise temperature
k = 1.38*10**-23    # Boltzmann's constant in J/K
c0 = 3e8            # speed of light

#### calculations ####
def compute_parameters(Gt, Pt, f, B, Gr, R):
    lam = c0 / f                                                        # wavelength in m
    Pn = k * T * B                                                      # noise power in W
    Pr = 10**(Gt / 10) * 10**(Gr/10) * Pt * (lam / (4 * np.pi * R))**2  # received power
    SNR = Pr / Pn                                                       # signal to noise ratio
    SNR_dB = 10 * np.log10(SNR)                                         # convert to dB
    H = B * np.log2(1 + SNR)                                            # channel capacity in bps
    return lam, Pn, Pr, SNR_dB, H

######## uplink calculations ########
#### adjustable parameters ####
Gt = float(input("Primary station tx antenna gain (dB): "))             # antenna gain in dBi
Pt = float(input("Transmit power (W): "))                               # transmit power in W
f = float(input("Transmit frequency (Hz): "))                           # frequency in Hz
B = float(input("Transmit signal bandwidth (Hz): "))                    # bandwidth in Hz
Gr = float(input("Repeater rx antenna gain (dB): "))                    # antenna gain in dBi
R = float(input("Distance from primary station to repeater station: ")) # distance in m 

lam, Pn, Pr, SNR_dB, H = compute_parameters(Gt, Pt, f, B, Gr, R)

#### output ####
print("Wavelength (m): ", lam)
print("Noise Power (W): ", Pn)
print("Received Power (W): ", Pr)
print("SNR (dB): ",  SNR_dB)
print("Channel Capacity (bps): ", H)


######## downlink calculations ########
#### adjustable parameters ####
Gt = float(input("Repeater station tx antenna gain (dB): "))                # antenna gain in dBi
Amp_Gain = float(input("Repeater gain (dB): "))                             # transmit power in W
f = float(input("Repeater new transmit frequency (Hz): "))                  # frequency in Hz
Gr = float(input("Secondary station rx antenna gain (dB): "))               # antenna gain in dBi
R = float(input("Distance from repeater station to secondary station: "))   # distance in m 

lam, Pn, Pr, SNR_dB, H = compute_parameters(Gt, Pt, f, B, Gr, R)

#### output ####
print("Wavelength (m): ", lam)
print("Noise Power (W): ", Pn)
print("Received Power (W): ", Pr)
print("SNR (dB): ",  SNR_dB)
print("Channel Capacity (bps): ", H)