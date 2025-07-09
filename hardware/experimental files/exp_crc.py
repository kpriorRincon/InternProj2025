# using the crc library
# https://pypi.org/project/crc/ for documentation
# by default use CCITT CRC-8 for testing

from crc import Calculator, Crc8

# create data to transmit
data = bytes([0, 1, 2, 3, 4, 5])
print("Data: ", data)

# calculate what to append to the end of the data
calculator = Calculator(Crc8.CCITT)
expected = calculator.checksum(data)
print("CRC-8: ", expected)

# append the crc-8 to the data
data = data + bytes([expected])
print("Data with CRC-8 appended: ", data)
