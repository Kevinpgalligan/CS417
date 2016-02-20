def floating_point_cardinality (significand_digits, base, exp_max, exp_min):
    """Returns how many numbers are representable given floating pt params.
    
    x = d1d2...d{significand_digits} * (base ** e)
        where   0 <= di < base, for every i
                exp_min <= e <= exp_max
    """
    
    # 2: each number can have 2 signs.
    # base - 1: first digit can be any value except 0 (normalisation).
    # base ** (significand_digits - 1): remaining digits can have any value.
    # (exp_max - exp_min + 1): exp_min <= exp <= exp_max.
    # + 1: representation of 0.
    return (2 * (base - 1) * (base ** (significand_digits - 1)) *
        (exp_max - exp_min + 1) + 1)
        
def abs_err (computed_val, correct_val):
    return abs(computed_val - correct_val)
    
def rel_err (computed_val, correct_val):
    return abs_err(computed_val, correct_val) / correct_val

def pct_err (computed_val, correct_val):
    return rel_err(computed_val, correct_val) * 100
    
def decimal_fraction_to_base2_mantissa (f, precision=10):
    """Requires: 0 <= f < 1."""
    bits = []
    
    k = 1
    while f != 0 and k <= precision:
        quotient = f / (1 / 2**k)
        if int(quotient) > 0:
            bits.append(1)
            f -= 1 / 2**k
        else:
            bits.append(0)
    
        k += 1
        
    return bits