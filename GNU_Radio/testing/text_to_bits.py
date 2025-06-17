# clear files before writing to it
open('bits_to_send.char', 'w').close()
open('bits_read_in.char', 'w').close()

# get message
message = input("Enter your message: ")
with open("bits_to_send.char", "wb") as f:
    f.write(message.encode('utf-8'))
