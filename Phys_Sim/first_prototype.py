import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater

#create all of the objects
sig_gen = Sig_Gen.SigGen()
receiver = Receiver.Receiver(sampling_rate=1e6, frequency=915e6)
repeater = Repeater.Repeater(sampling_rate=1e6, frequency=915e6)
user_input = input("Enter a message to be sent: ")
message_bits = sig_gen.set_message(user_input)
sig_gen.generate_qpsk()