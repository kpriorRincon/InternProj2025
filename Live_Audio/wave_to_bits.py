#Experimental code to get bits from wav file.
import wave as wv

# how many samples to read at a time
CHUNK = 1024

with wv.open('output.wav', 'rb') as wave_file:
    # Get file parameters
    d = wave_file.readframes(CHUNK)
    print(d)