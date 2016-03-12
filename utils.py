import numpy as np

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