import numpy as np

# Load file
raw_data = np.fromfile("bits_read_in.bin", dtype=np.complex64)
print("Raw data loaded from file:\n", raw_data)

# Convert raw data to list
complex_data = raw_data.tolist()
print("First 32 items received:", complex_data[:32])
print("Total items received: ", len(complex_data))

# Format without parentheses around each number
complex_numbers = f"({', '.join(f'{num.real:g}{num.imag:+g}j' for num in complex_data)})"

# Save to text file
with open("bits_read_in.txt", "w") as outfile:
    outfile.write(complex_numbers)
