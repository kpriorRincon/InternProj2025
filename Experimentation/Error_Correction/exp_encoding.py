import reedsolo as rs

#### Transmit Side ####
# Initialize RS codec with GF(2^8) primitive polynomial
rs.init_tables(0x11d)

# Define message and ECC size
nsym = 12  # Number of ECC symbols
msg = b"Hello, Reed-Solomon!"  # Sample message
print(f"Original message: {msg}")
print(f"Message length: {len(msg)} bytes")
print(f"ECC symbols: {nsym}")

# Generate the generator polynomial for nsym ECC symbols
gen = rs.rs_generator_poly(nsym)  # Generate polynomial for nsym symbols
print(f"Generator polynomial degree: {len(gen)-1}")

# Encode the message
mesecc = rs.rs_encode_msg(msg, nsym, gen=gen)
print(f"Encoded message length: {len(mesecc)} bytes")
print(f"Encoded message: {mesecc}")

#### Channel Effecs ####
# Introduce errors to test error correction
print("\n--- Introducing errors ---")
mesecc_corrupted = bytearray(mesecc)  # Make a mutable copy
mesecc_corrupted[1] = 0  # Corrupt first data byte
mesecc_corrupted[5] = 255  # Corrupt another byte
print(f"Corrupted message: {bytes(mesecc_corrupted)}")

#### Receive Side ####
# Decode and correct the message
rmes, recc, errata_pos = rs.rs_correct_msg(mesecc_corrupted, nsym, gen=gen)
print(f"\n--- Correction results ---")
print(f"Corrected message: {rmes}")
print(f"Corrected ECC: {recc}")
print(f"Error positions: {errata_pos}")
print(f"Successfully corrected: {rmes == msg}")