import numpy as np

from math import log
from random import randint

import utils
import lin_systems as ls

DEFAULT_FACTOR_BASE_SIZE = 5

def index_calculate (y, g, p, factor_base_size=DEFAULT_FACTOR_BASE_SIZE):
    """Find x to satisfy y = g^x (mod p)."""
    factor_base = get_factor_base(factor_base_size)
    factor_base_logs = get_factor_base_logs(factor_base, g, p)
    
    e = 1
    factorisation = None
    while factorisation is None:
        factorisation = factorise_using_factor_base((y * (g ** e)) % p)
        
        e += 1
    
    x = - e
    for p, e in factorisation:
        x = (x + e * factor_base_logs[p]) % (p - 1)
    
    return x
    
def get_factor_base (n):
    factor_base = [2]
    
    bound = int(2 * n * log(n))
    
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
    factorised_powers_of_g = []
    
    while len(factorised_powers_of_g) < len(factor_base):
        e = randint(1, p - 2)
    
        ge = (g ** e) % p
        
        factorisation = factorise_using_factor_base(ge, factor_base)
        if factorisation is not None:
            factorised_powers_of_g.append((e, factorisation))
    
    lin_sys = np.zeros((len(factor_base), len(factor_base) + 1))
    for i, (e, factorisation) in enumerate(factorised_powers_of_g):
        lin_sys[i, -1] = e
        for j, p in enumerate(factor_base):
            lin_sys[i, j] = factorisation.get(p, 0)
            
    print(lin_sys)
    
    factor_base_log_g_solns = ls.ge_mod_n(lin_sys, p - 1)
    
    factor_base_logs = dict(zip(factor_base, factor_base_log_g_solns))
    
    return factor_base_logs