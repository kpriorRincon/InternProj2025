# imports
import numpy as np

#################################################
#
#   Author: Trevor Wiseman
#
#################################################

#### constant globals ####
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
    H = B * np.log2(1 + SNR)                                            # channel capacity in bits/s     
    return lam, Pn, Pr, SNR_dB, H

def arrival_time(R):
    return R/c0 # time taken for signal to travel distance R

def frequency_doppler_shift(v, fc, psi):
    lam = c0 / fc                       # wavelength of the carrier frequency in m
    fmax = v / lam                      # doppler shift frequency in Hz
    return fmax*np.cos(psi) + fc   # received frequency including the doppler shift