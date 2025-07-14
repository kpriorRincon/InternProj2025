# This example demonstrates
# 1. Using PyVISA (https://pyvisa.readthedocs.io/) to connect to Spike
# 2. Putting the device into non-continuous mode
# 3. Performing and waiting for a sweep to complete

import pyvisa

# Get the VISA resource manager
rm = pyvisa.ResourceManager()

# Open a session to the Spike software, Spike must be running at this point
inst = rm.open_resource('TCPIP0::localhost::5025::SOCKET')

# For SOCKET programming, we want to tell VISA to use a terminating character
#   to end a read and write operation.
inst.read_termination = '\n'
inst.write_termination = '\n'

# Set the measurement mode to sweep
inst.write("INSTRUMENT:SELECT SA")
# Disable continuous meausurement operation
inst.write("INIT:CONT OFF")

# Configure a 20MHz span sweep at 1GHz
# Set the RBW/VBW to auto
inst.write("SENS:BAND:RES:AUTO ON; :BAND:VID:AUTO ON; :BAND:SHAPE FLATTOP")
# Center/span
inst.write("SENS:FREQ:SPAN 20MHZ; CENT 1GHZ")
# Reference level/Div
inst.write("SENS:POW:RF:RLEV -20DBM; PDIV 10")
# Peak detector
inst.write("SENS:SWE:DET:FUNC MINMAX; UNIT POWER")

# Configure the trace. Ensures trace 1 is active and enabled for clear-and-write.
# These commands are not required to be sent everytime, this is for illustrative purposes only.
inst.write("TRAC:SEL 1") # Select trace 1
inst.write("TRAC:TYPE WRITE") # Set clear and write mode
inst.write("TRAC:UPD ON") # Set update state to on
inst.write("TRAC:DISP ON") # Set un-hidden

# Trigger a sweep, and wait for it to complete
inst.query(":INIT; *OPC?")

# Sweep data is returned as comma separated values
data = inst.query("TRACE:DATA?")

# Split the returned string into a list
points = [float(x) for x in data.split(',')]

# Query information needed to know what frequency each point in the sweep refers to
start_freq = float(inst.query("TRACE:XSTART?"))
bin_size = float(inst.query("TRACE:XINC?"))

# Find the peak point in the sweep
peak_val = max(points)
peak_idx = points.index(peak_val)
peak_freq = start_freq + peak_idx * bin_size

# Print out peak information
print (f"Peak Freq {peak_freq / 1.0e6} MHz, Peak Ampl {peak_val} dBm\n")

# Done
inst.close()
