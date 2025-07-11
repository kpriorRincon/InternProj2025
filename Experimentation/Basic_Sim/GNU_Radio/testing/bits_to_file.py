def bits_to_bytes(bits):
    """
    Take a bitstream and return a file of bytes represented as integers

    Parameters:
    bits (string): Bitstream

    Returns:
    bytes.txt: A text file of bytes represented as integers. Each byte is on a new line. 

    """
    bytes = [bits[i:i+8] for i in range(0,len(bits),8)]
    bytes_int = [int(byte,2) for byte in bytes]
    with open("bytes.txt", "w") as outfile:
        for byte in bytes_int:
            outfile.write(f"{byte}\n")