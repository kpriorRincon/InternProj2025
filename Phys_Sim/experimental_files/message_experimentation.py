string = input("Enter a message\n")

binary = ''.join(format(ord(x), '08b') for x in string)
print(binary)

message = ''.join(chr(int(binary[i*8:i*8+8],2)) for i in range(len(binary)//8))
print(message)