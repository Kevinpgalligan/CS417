import numpy as np

from random import randint

import utils
import lin_systems as ls

DEFAULT_FACTOR_BASE_SIZE = 5

# Proportion of numbers you fail to factorise with factor base before
# giving up and increasing its size.
FACTOR_BASE_INCREASE_THRESHOLD = 0.001

# How many times you can try to add a new equation to the log system
# before throwing it out and starting from scratch.
LINEAR_DEPENDENCE_LIMIT = 100

def index_calculate (y, g, p, factor_base_size=DEFAULT_FACTOR_BASE_SIZE):
    """Find x to satisfy y = g^x (mod p)."""
    largest_fb_size = utils.approximate_num_possible_prime_divisors(p)
    
    factor_base_logs = None
    while factor_base_logs is None:
        factor_base = get_factor_base(factor_base_size)
        factor_base_logs = get_factor_base_logs(factor_base, g, p)
        
        factor_base_size = min(2 * factor_base_size, largest_fb_size)
    
    factorisation = None
    while factorisation is None:
        e = randint(1, p - 2)
        factorisation = factorise_using_factor_base((y * (g ** e)) % p,
            factor_base)
    
    x = - e
    for prime, exp in factorisation.items():
        x = (x + exp * factor_base_logs[prime]) % (p - 1)
    
    return x
    
def get_factor_base (n):
    factor_base = [2]
    
    bound = 2 * utils.nth_prime_approximation(n)
    
    sieve = [True for _ in range(bound)]
    i = 3
    while len(factor_base) < n:
        if sieve[i]:
            factor_base.append(i)
        
            for j in range(i + i, len(sieve), i):
                sieve[j] = False
        
        i += 2
    
    return factor_base
    
def factorise_using_factor_base (x, factor_base):
    factorisation = {}

    for p in factor_base:
        e = 0
        while x % p == 0:
            x = x // p
            e += 1
        
        if e != 0:
            factorisation[p] = e
        if x == 1:
            break
    
    return factorisation if x == 1 else None

def get_factor_base_logs (factor_base, g, p):
    log_equations = []
    
    successful_factorisation_attempts = 1
    total_factorisation_attempts = 1
    
    while len(log_equations) < len(factor_base):
        e = randint(1, p - 2)
    
        ge = (g ** e) % p
        
        factorisation = factorise_using_factor_base(ge, factor_base)
        if factorisation is not None:
            successful_factorisation_attempts += 1
        
            equation = []
            for q in factor_base:
                equation.append(factorisation.get(q, 0))
            equation.append(e)
            
            if utils.is_linearly_independent_mod_n(equation,
                    log_equations, p - 1):
                log_equations.append(equation)
                linear_dependence_count = 0
            elif linear_dependence_count > LINEAR_DEPENDENCE_LIMIT:
                log_equations = []
            else:
                linear_dependence_count += 1
        
        total_factorisation_attempts += 1
        
        if (successful_factorisation_attempts / total_factorisation_attempts <
                FACTOR_BASE_INCREASE_THRESHOLD):
            return None
    
    lin_sys = np.array(log_equations)

    factor_base_log_g_solns = ls.ge_mod_n(lin_sys, p - 1)
    
    factor_base_logs = dict(zip(factor_base, factor_base_log_g_solns))
    
    return factor_base_logs

def main ():
    print(index_calculate(13, 6, 229))
    print(index_calculate(20, 5, 503))

if __name__ == '__main__':
    main()