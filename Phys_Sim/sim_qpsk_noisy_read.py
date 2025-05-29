# imports
import numpy as np
import matplotlib.pyplot as plt

#################################################
#
#   Author: Trevor Wiseman
#
#################################################

######## Functions ########

# generate random QPSK symbols
def random_symbol_generator(num_symbols=100):
    # generate random QPSK symbols
    x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
    x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
    x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
    x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols
    return x_symbols

# add noise to the symbols
def noise_adder(x_symbols, noise_power=0.1, num_symbols=100):
    n = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))/np.sqrt(2) # AWGN with unity power
    phase_noise = np.random.randn(len(x_symbols)) * 0.1 # adjust multiplier for "strength" of phase noise
    r = x_symbols * np.exp(1j*phase_noise) + n * np.sqrt(noise_power)
    return r

# read the bits from the noisy QPSK symbols
def bit_reader(r):
    bits = np.zeros((len(r), 2), dtype=int) # an array of bits
    for i in range(len(r)):
        angle = np.angle(r[i], deg=True) # get the angle of the symbol
        print("angle: ", angle)
        if (angle >= 0 and angle < 90): # 45 degrees
            bits[i][0] = 0
            bits[i][1] = 0
        elif (angle >= 90 and angle < 180): # 135 degrees
            bits[i][0] = 0
            bits[i][1] = 1
        elif (angle >= -90 and angle < -180): # 225 degrees
            bits[i][0] = 1
            bits[i][1] = 1
        elif (angle >= -90 and angle < 0): # 315 degrees
            bits[i][0] = 1
            bits[i][1] = 0
    
    return bits

######## Test Code ########
# x_symbols = random_symbol_generator() # generate QPSK symbols
# r = noise_adder(x_symbols) # add noise to the symbols
# bits = bit_reader(r) # convert symbols to bits

# # print the symbols and bits
# print("Print QPSK symbols with noise:")
# print(r)
# print("Print QPSK symbols:")
# print(bits)

# # Plot the noisy QPSK symbols
# plt.plot(np.real(r), np.imag(r), '.')
# plt.grid(True)
# plt.show()