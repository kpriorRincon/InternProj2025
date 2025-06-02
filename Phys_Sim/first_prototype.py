import Sig_Gen as Sig_Gen
import Receiver as Receiver
import Repeater as Repeater
from sim_qpsk_noisy_demod import sample_read_output

#create all of the objects
symbol_rate = 10e6
f_carrier = 910e6
fs = 4e9 #sample frequency
desired_f = 960e6


def main():
    sig_gen = Sig_Gen.SigGen()
    sig_gen.freq = f_carrier
    sig_gen.sample_rate = fs
    sig_gen.symbol_rate = symbol_rate 

    repeater = Repeater.Repeater(desired_frequency=desired_f, sampling_frequency=fs)

    user_input = input("Enter a message to be sent: ")
    message_bits = sig_gen.message_to_bits(user_input)

    print(message_bits)
    t, qpsk, lines, symbols = sig_gen.generate_qpsk(message_bits)
    
    analytic_signal, bits = sample_read_output(qpsk, fs, symbol_rate)
    print(f"After generation: {bits}")

    qpsk_mixed = repeater.mix(qpsk, sig_gen.freq, t)
    analytic_signal, bits = sample_read_output(qpsk_mixed, fs, symbol_rate)
    print(f"After mixing: {bits}")

    qpsk_filtered = repeater.filter(desired_f + 20e6, qpsk_mixed, order=5)
    
    analytic_signal, bits = sample_read_output(qpsk_filtered, fs, symbol_rate)
    print(f"After filter: {bits}")

    qpsk_amp = repeater.amplify(gain=2, input_signal=qpsk_filtered)
    
    analytic_signal, bits = sample_read_output(qpsk_amp, fs, symbol_rate)
    print(f"After amp: {bits}")


if __name__ == "__main__":
    main()