# Runs all zmq_ scripts for implementation into the GUI

import subprocess

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