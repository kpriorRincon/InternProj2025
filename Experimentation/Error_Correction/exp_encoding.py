import reedsolo as rs

# initialize RS codec
rs.init_tables(0x11d)

# define message and ecc size
n = 255                         # length of total message+ecc
nsym = 12                       # length of ecc
msg = b"Hello, Reed-Solomon!"   # sample message
n = len(msg) + nsym
gen = rs.rs_generator_poly(n)
