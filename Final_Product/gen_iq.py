from config import *
import transmit_processing as tp
import sys
#get the message from sys argv
if len(sys.argv)< 2:
    print("Usage: python gen_iq.py \"your message here\"")
    sys.exit(1)
message = sys.argv[1]

#create transmit processing object then run the work function
tp = tp.transmit_processing(int(SAMPLE_RATE/SYMB_RATE), SAMPLE_RATE)
tp.work(message, BETA, NUMTAPS)