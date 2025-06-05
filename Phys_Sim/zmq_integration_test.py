# Runs all zmq_ scripts for implementation into the GUI

import subprocess
import pickle

scripts = [
    ['python', 'zmq_transmitter.py'],
    ['python', 'zmq_repeater.py'],
    ['python', 'zmq_receiver.py'],
    ['python', 'zmq_controller.py']
           ]

processes = []

for script in scripts:
    process = subprocess.Popen(script)
    processes.append(process)

for process in processes:
    process.wait()

for process in processes:
    process.terminate()

with open('controller_data.pkl', 'rb') as infile:
    data = pickle.load(infile)

t = data['time']
tx_signal = data['transmitter signal']
tx_symbols = data['transmitter symbols']
tx_message_binary = data['transmitter message in binary']
rep_incoming_signal = data['repeater incoming signal']
rep_mixed_signal = data['repeater mixed signal']
rep_filtered_signal = data['repeater filtered signal']
rep_outgoing_signal = data['repeater outgoing signal']
rx_message_binary = data['receiver message in binary']
rx_recovered_message = data['receiver recovered message']
rx_incoming_signal = data['receiver incoming signal']
rx_filtered_signal = data['receiver filtered signal']