# import numpy as np

# def linear_feedback_shift_register():

#     return m1, m2

# def gold_code_generator(m1, m2):
#     m1_shifted = m1 >> 1
#     xor_sequence = m1_shifted ^ m2
#     return xor_sequence

# def main():
#     m1 = np.array([0, 0, 0, 1, 1, 1, 1, 0])
#     m2 = np.array([1, 0, 1, 1, 0, 1, 0, 0])

#     gold_code = gold_code_generator(m1, m2)
#     print("Gold Code: ", gold_code)

# main()

import numpy as np

def generate_msequence(taps, length, initial_state=None):
    """
    Generate a maximum-length sequence (m-sequence) using a linear feedback shift register.
    
    Args:
        taps: List of tap positions (1-indexed) for the feedback polynomial
        length: Length of the sequence to generate
        initial_state: Initial state of the shift register (if None, uses all ones)
    
    Returns:
        numpy array of the m-sequence
    """
    # Number of stages in the shift register
    n_stages = max(taps)
    
    # Initialize shift register
    if initial_state is None:
        shift_register = np.ones(n_stages, dtype=int)
    else:
        shift_register = np.array(initial_state, dtype=int)
    
    sequence = []
    
    for _ in range(length):
        # Output current bit (typically the last stage)
        output_bit = shift_register[-1]
        sequence.append(output_bit)
        
        # Calculate feedback bit by XORing tapped positions
        feedback = 0
        for tap in taps:
            feedback ^= shift_register[tap - 1]  # Convert to 0-indexed
        
        # Shift register and insert feedback
        shift_register = np.roll(shift_register, 1)
        shift_register[0] = feedback
    
    return np.array(sequence)

def generate_gold_codes(m, preferred_pairs=None):
    """
    Generate Gold codes for a given m (degree of polynomial).
    
    Args:
        m: Degree of the primitive polynomial
        preferred_pairs: Dictionary of preferred pairs for different m values
    
    Returns:
        List of Gold code sequences
    """
    # Default preferred pairs for common values of m
    if preferred_pairs is None:
        preferred_pairs = {
            3: ([3, 2], [3, 1, 2, 1]),  # For m=3
            4: ([4, 3], [4, 1]),         # For m=4
            5: ([5, 2], [5, 4, 3, 2]),  # For m=5
            6: ([6, 1], [6, 5, 2, 1]),  # For m=6
            7: ([7, 3], [7, 3, 2, 1]),  # For m=7
            10: ([10, 3], [10, 8, 3, 2]) # For m=10 (GPS)
        }
    
    if m not in preferred_pairs:
        raise ValueError(f"No preferred pair defined for m={m}")
    
    taps1, taps2 = preferred_pairs[m]
    sequence_length = 2**m - 1
    
    # Generate the two preferred m-sequences
    seq1 = generate_msequence(taps1, sequence_length)
    seq2 = generate_msequence(taps2, sequence_length)
    
    # Generate Gold codes
    gold_codes = []
    
    # Add the two original sequences
    gold_codes.append(seq1)
    gold_codes.append(seq2)
    
    # Generate all cyclic shifts of seq2 and XOR with seq1
    for shift in range(1, sequence_length):
        shifted_seq2 = np.roll(seq2, shift)
        gold_code = seq1 ^ shifted_seq2
        gold_codes.append(gold_code)
    
    return gold_codes

def bipolar_representation(binary_sequence):
    """Convert binary sequence (0,1) to bipolar representation (-1,+1)."""
    return 2 * binary_sequence - 1

def calculate_autocorrelation(sequence):
    """Calculate the autocorrelation function of a sequence."""
    n = len(sequence)
    bipolar_seq = bipolar_representation(sequence)
    autocorr = np.correlate(bipolar_seq, bipolar_seq, mode='full')
    return autocorr[n-1:]  # Return only non-negative lags

def calculate_cross_correlation(seq1, seq2):
    """Calculate cross-correlation between two sequences."""
    bipolar_seq1 = bipolar_representation(seq1)
    bipolar_seq2 = bipolar_representation(seq2)
    return np.correlate(bipolar_seq1, bipolar_seq2, mode='full')

# Example usage
if __name__ == "__main__":
    # Generate Gold codes for m=5
    m = 5
    print(f"Generating Gold codes for m={m}")
    print(f"Sequence length: {2**m - 1}")
    
    gold_codes = generate_gold_codes(m)
    print(f"Number of Gold codes generated: {len(gold_codes)}")
    
    # Display first few codes
    print("\nFirst 5 Gold codes (binary representation):")
    for i, code in enumerate(gold_codes[:5]):
        print(f"Code {i+1}: {code}")
    
    # Display in bipolar representation
    print("\nFirst 3 Gold codes (bipolar representation -1/+1):")
    for i, code in enumerate(gold_codes[:3]):
        bipolar_code = bipolar_representation(code)
        print(f"Code {i+1}: {bipolar_code}")
    
    # Calculate and display autocorrelation properties
    print("\nAutocorrelation properties:")
    for i, code in enumerate(gold_codes[:3]):
        autocorr = calculate_autocorrelation(code)
        max_sidelobe = max(abs(autocorr[1:]))  # Exclude zero lag
        print(f"Code {i+1} - Max autocorr sidelobe: {max_sidelobe}")
    
    # Calculate cross-correlation between first two codes
    if len(gold_codes) >= 2:
        cross_corr = calculate_cross_correlation(gold_codes[0], gold_codes[1])
        max_cross_corr = max(abs(cross_corr))
        print(f"\nMax cross-correlation between Code 1 and Code 2: {max_cross_corr}")
    
    # Example for GPS-like application (m=10)
    print(f"\n{'='*50}")
    print("GPS-like Gold codes (m=10) - showing properties only")
    try:
        gps_codes = generate_gold_codes(10)
        print(f"Generated {len(gps_codes)} codes of length {len(gps_codes[0])}")
        
        # Show autocorrelation properties for first code
        autocorr = calculate_autocorrelation(gps_codes[0])
        max_sidelobe = max(abs(autocorr[1:]))
        print(f"Max autocorrelation sidelobe: {max_sidelobe}")
        
    except Exception as e:
        print(f"Error generating GPS codes: {e}")