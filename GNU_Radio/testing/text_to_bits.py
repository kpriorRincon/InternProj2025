# clear files before writing to it
open('bits_to_send.txt', 'w').close()
open('bits_read_in.txt', 'w').close()

# get message
message = input("Enter your message: ")
with open("bits_to_send.txt", "wb") as f:
    f.write(message.encode('utf-8'))
