import bits_to_file as btf

# clear files before writing to it
open('bits_to_send.txt', 'w').close()
open('bits_read_in.txt', 'w').close()

# get message
message = input("Enter your message: ")
print("Transmitted message length: ", len(message))

# convert to bit string
bits_message = ''.join(format(ord(c), '08b') for c in message)
print("Transmitted bits length: ", len(bits_message))

# converr to list of bits
bits_list = [int(bit) for bit in bits_message]

# write bits to binary file
with open('bits_to_send.bin', 'wb') as f:
    for bit in bits_list:
        f.write(bytes([bit]))

print("First 32 bits sent:", bits_list[:32])
