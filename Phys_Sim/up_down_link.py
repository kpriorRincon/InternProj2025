# imports
import numpy as np
import matplotlib.pyplot as plt

#### adjustable parameters ####
Gt = int(input("Transmitter gain (dB): "))      # antenna gain in dBi
Pt = int(input("Transmit power (W): "))   # transmit power in W
f = int(input("Transmit frequency (Hz): "))   # frequency in Hz
B = int(input("Transmit signal bandwidth (Hz): "))     # bandwidth in Hz
Gr = int(input("Receiver gain (dB): "))      # antenna gain in dBi
R = int(input("Distance from ground station to space station: ")) # distance in m 

#### predefined parameters ####
T = 290             # noise temperature
k = 1.38*10**-23    # Boltzmann's constant in J/K
c0 = 3e8            # speed of light

#### calculations ####
lam = c0 / f                                                        # wavelength in m
Pn = k * T * B                                                      # noise pwower in W
Pr = 10**(Gt / 10) * 10**(Gr/10) * Pt * (lam / (4 * np.pi * R))**2  # received power
SNR = Pr / Pn                                                       # signal to noise ratio
SNR_dB = 10 * np.log10(SNR)                                         # convert to dB
H = B * np.log2(1 + SNR)                                            # channel capacity in bps
# Multipath Fading

#### output ####
print("Wavelength (m): ", lam)
print("Noise Power (W): ", Pn)
print("Received Power (W): ", Pr)
print("SNR (dB): ",  SNR_dB)
print("Channel Capacity (bps): ", H)