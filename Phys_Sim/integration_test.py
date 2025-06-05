import subprocess

scripts = [
    ['python', 'zmq_controller.py'],
    ['python', 'zmq_transmitter.py'],
    ['python', 'zmq_repeater.py'],
    ['python', 'zmq_receiver.py']
           ]

processes = []

for script in scripts:
    process = subprocess.Popen(script)
    processes.append(process)

for process in processes:
    process.wait()