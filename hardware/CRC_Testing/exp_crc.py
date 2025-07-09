# using the crc library
# https://pypi.org/project/crc/ for documentation
# by default use CCITT CRC-8 for testing

from crc import Calculator, Crc8

# create data to transmit
data = bytes([0, 1, 2, 3, 4, 5])
print("Data: ", data)

# calculate what to append to the end of the data
calculator = Calculator(Crc8.CCITT)
code = calculator.checksum(data)
print("CRC-8: ", code)

# append the crc-8 to the data
data = data + bytes([code])
print("Data with CRC-8 appended: ", data)

# error check
check = calculator.checksum(data)
print("Remainder: ", check)
if check == 0:
    print("Data is valid...\nStart Processing...")
else:
    print("Data is invalid...\nAborting...")