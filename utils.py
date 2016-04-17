import numpy as np

from math import sqrt

def read_matrix (s, entry_separator=" ", row_separator=":", dtype=float):
    """Given string, parses matrix."""
    rows = s.split(row_separator)
    matrix = [[dtype(n) for n in row.split(entry_separator)] for row in rows]
    
    return np.array(matrix, dtype=dtype)

def is_matrix (arr):
    """Checks that all rows in an array have the same # of entries."""
    if len(arr) == 0: return True
    
    row_length = len(arr[0])
    i = 1
    while i < len(arr):
        if row_length != len(arr[i]):
            return False
        
        i += 1
        
    return True
    
def generate_hilbert_matrix (rows, columns=None):
    # TBD: look up numpy method for appending column.
    # (this doesn't generate an augmented matrix).
    if columns is None:
        columns = rows
        
    return np.array([[1/n for n in range(row, row + columns)] for
        row in range(1, rows + 1)])

def mod_pow (x, e, n):
    """Fast exponentiation: x^e (mod n)."""
    a = x
    b = 1
    
    while e:
        if e % 2 == 0:
            a = (a * a) % n
            e = e // 2
        else:
            b = (a * b) % n
            e -= 1
    
    return b

def gcd (a, b):
    while b != 0:
       a, b = b, a % b
       
    return a

def extended_gcd (a, b):
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q, r = b//a, b%a
        m, n = x-u*q, y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
    return b, x, y  

def modinv (a, m):
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        return None
    else:
        return x % m

def factorise (n):
    """...by trial division."""
    factorisation = []
    
    e = 0
    while n % 2 == 0:
        n = n // 2
        e += 1
    if e != 0:
        factorisation.append((2, e))
    
    sqrt_n = int(sqrt(n))
    
    p = 3
    while n != 1:
        if p > sqrt_n:
            factorisation.append((n, 1))
            break
    
        e = 0
        while n % p == 0:
            n = n // p
            e += 1
        
        if e != 0:
            factorisation.append((p, e))
        
        p += 2
    
    return factorisation
    
def chinese_remainder_theorem (solution_pairs):
    """
    `solution_pairs` - for x=a1 mod m1, x=a2 mod m2, ..., this list should be
        [(a1, m1), (a2, m2), ...]
    """
    M = 1
    for _, m in solution_pairs:
        M *= m
        
    x = 0
    for a, m in solution_pairs:
        Mi = M // m
        x = (x + a * Mi * modinv(Mi, m)) % M
    
    return x