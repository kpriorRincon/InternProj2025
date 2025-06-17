import bits_to_file as btf

# clear files before writing to it
open('bits_to_send.txt', 'w').close()
open('bits_read_in.txt', 'w').close()

# get message
message = input("Enter your message: ")

# convert to bit string
bits_message = ''.join(format(ord(c), '08b') for c in message)

# convert bits to bytes and write to file
btf.bits_to_bytes(bits_message)

# # write bits to file
# with open("bits_to_send.txt", "w") as f:
#     f.write(bits_message)
