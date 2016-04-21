import numpy as np

import utils

class PivotNotFoundException (Exception):
    def __init__ (self, pivot_row, matrix, *args, **kwargs):
        Exception.__init__(self, "Couldn't find pivot at " +
            "index ({0}, {0}) for the ".format(pivot_row) +
            "below matrix:\n{}".format(matrix),
            *args, **kwargs)

def swap_rows (matrix, i, j):
    matrix[[i, j], :] = matrix[[j, i], :]

def split_aug_matrix (aug_matrix):
    """Takes augmented matrix representing Ax=b, returns (A, b)."""
    A = aug_matrix[:, :-1]
    b = aug_matrix[:, -1]
    
    return A, b

def augment_matrix (A, b):
    """Add column b to matrix A."""
    return np.c_[A, b]

def get_upper_triangular_form (matrix, pivot_getter):
    """Returns a copy of `matrix` in upper-triangular form.
    
    `pivot_getter` - function that takes matrix and row index as parameters.
        Finds pivot in column below matrix[`row`,`row].
    """
    new_matrix = np.copy(matrix)
    to_upper_triangular_form(new_matrix, pivot_getter)
    return new_matrix
    
def to_upper_triangular_form (matrix, pivot_getter):
    """In-place conversion of matrix to upper-triangular form.
    
    `pivot_getter` - function that takes matrix and row index as parameters.
        Finds pivot in column below matrix[`row`,`row].
    """
    row = 0
    while row < len(matrix):
        pivot_row = pivot_getter(matrix, row)
        
        if pivot_row is None:
            raise PivotNotFoundException(row, matrix)
        elif pivot_row != row:
            # Have to swap rows to get pivot in place.
            swap_rows(matrix, row, pivot_row)
        
        # Now zero the values below the pivot in that column.
        for i in range(row + 1, len(matrix)):
            if matrix[i, row] != 0:
                scale_factor = matrix[i, row] / matrix[row, row]
                matrix[i] -= scale_factor * matrix[row]
        
        row += 1

def get_pivot_row (matrix, row):
    """Returns index of row containing pivot for `row`-th row.
    
    SIMPLE algorithm to find `pivot_row`:
    pivot_row = first `i` such that matrix[`i`, `row`] != 0, for 
    i = `row`, `row`+1, `row`+2, ...
    """
    pivot_row = None
    
    i = row
    while i < len(matrix):
        if matrix[i, row] != 0:
            pivot_row = i
            break
            
        i += 1
        
    return pivot_row

def _generic_get_partial_pivoting_row (matrix, row, pivot_evaluator):
    pivot_row = None
    best_pivot_val = None
    
    i = row
    while i < len(matrix):
        if matrix[i, row] != 0:
            pivot_val = pivot_evaluator(matrix, i)
            if pivot_row is None or pivot_val> best_pivot_val:
                pivot_row = i
                best_pivot_val = pivot_val
        
        i += 1
    
    return pivot_row

def _partial_pivot_eval (matrix, row):
    # Simply return the value of the potential pivot.
    return matrix[row, row]

def get_partial_pivoting_row (matrix, row):
    return _generic_get_partial_pivoting_row(matrix, row, _partial_pivot_eval)

def _scaled_partial_pivot_eval (matrix, row):
    # QUESTION: include the pivot itself in max row val? (unclear from notes).
    # If not, then divide by max(matrix[row, row+1:]).
    return matrix[row, row] / max(matrix[row, row:])

def get_scaled_partial_pivoting_row (matrix, row):
    return _generic_get_partial_pivoting_row(matrix, row,
        _scaled_partial_pivot_eval)

def back_substitute (matrix):
    """Returns solutions for upper triangular form augmented matrix."""
    solutions = []
    
    i = len(matrix) - 1
    while i >= 0:
        solution = (matrix[i, -1] -
            np.sum(matrix[i, i+1:-1] * solutions)) / matrix[i, i]
        solutions.insert(0, solution)
        
        i -= 1
        
    return solutions

def forward_substitute (matrix):
    """Returns solutions for lower triangular form augmented matrix."""
    solutions = []
    
    i = 0
    while i < len(matrix):
        solution = (matrix[i, -1] -
            np.sum(matrix[i, :i] * solutions)) / matrix[i, i]
        solutions.append(solution)
        
        i += 1
        
    return solutions

def gaussian_elimination (aug_matrix, pivot_getter):
    upper_triangular_form = get_upper_triangular_form(aug_matrix,
        pivot_getter)
    return back_substitute(upper_triangular_form)

def simple_ge (aug_matrix):
    return gaussian_elimination(aug_matrix, get_pivot_row)
    
def partial_pivoting_ge (aug_matrix):
    return gaussian_elimination(aug_matrix, get_partial_pivoting_row)
    
def partial_pivoting_with_scaling_ge (aug_matrix):
    return gaussian_elimination(aug_matrix, get_scaled_partial_pivoting_row)

def lu_factorise (matrix, pivot_getter=get_scaled_partial_pivoting_row):
    permutation_matrix = np.identity(len(matrix))
    A = np.copy(matrix)
    
    row = 0
    while row < len(A):
        pivot_row = pivot_getter(A, row)
        
        if pivot_row is None:
            raise PivotNotFoundException(row, matrix)
        elif pivot_row != row:
            swap_rows(A, row, pivot_row)
            swap_rows(permutation_matrix, row, pivot_row)
        
        for i in range(row + 1, len(A)):
            if A[i, row] != 0:
                scale_factor = A[i, row] / A[row, row]
                A[i, row+1:] -= scale_factor * A[row, row+1:]
                A[i, row] = scale_factor
        
        row += 1
    
    L = np.eye(len(matrix)) + np.tril(A, -1)
    U = np.triu(A)
    
    return L, U, permutation_matrix
    
def lu_method (aug_matrix, pivot_getter=get_scaled_partial_pivoting_row):
    A, b = split_aug_matrix(aug_matrix)
    
    L, U, permutation_matrix = lu_factorise(A, pivot_getter)
    b = np.dot(permutation_matrix, b[:, np.newaxis])
    
    # Given Ax = LUx = L(Ux) = b.
    # Let y = Ux.
    # Then solve Ly = b for y.
    # This gives Ux = y. Solve for x.
    L = augment_matrix(L, b)
    y = forward_substitute(L)
    
    U = augment_matrix(U, y)
    x = back_substitute(U)
    
    return x

def get_upper_triangular_mod_n (aug_matrix, n):
    new_matrix = np.copy(aug_matrix)
    to_upper_triangular_mod_n(new_matrix, n)
    return new_matrix
    
def get_pivot_row_mod_n (aug_matrix, row, n):
    for i in range(row, len(aug_matrix)):
        inverse = utils.modinv(aug_matrix[i, row], n)
        if aug_matrix[i, row] != 0 and inverse is not None:
            return i, inverse
    
    return None, None

def to_upper_triangular_mod_n (aug_matrix, n):
    for row in range(len(aug_matrix[0]) - 1):
        pivot_row, pivot_inv = get_pivot_row_mod_n(aug_matrix, row, n)
        
        if pivot_row is None:
            print(aug_matrix[40])
            raise PivotNotFoundException(row, aug_matrix)
        elif pivot_row != row:
            swap_rows(aug_matrix, pivot_row, row)
        
        for i in range(row + 1, len(aug_matrix)):
            scale_factor = (pivot_inv * aug_matrix[i, row]) % n
            for j in range(row, len(aug_matrix[i])):
                aug_matrix[i, j] = (aug_matrix[i, j] - scale_factor *
                    aug_matrix[row, j]) % n

def back_substitute_mod_n (aug_matrix, n):
    solns = []

    for i in range(len(aug_matrix[0]) - 2, -1, -1):
        xi = ((aug_matrix[i, -1] - sum(solns * aug_matrix[i, i+1:-1])) *
            utils.modinv(aug_matrix[i, i], n)) % n
            
        solns.insert(0, xi)
    
    return solns

def _ge_solve_mod_n (aug_matrix, pp):
    upper_t_form = get_upper_triangular_mod_n(aug_matrix, pp)
    return back_substitute_mod_n(upper_t_form, pp)

def ge_mod_n (aug_matrix, n, n_factorisation=None):
    if n_factorisation is None:
        n_factorisation = utils.factorise(n)

    solutions = []

    prime_powers = [p ** e for p, e in n_factorisation]
    
    solutions_for_prime_power = []
    for pp in prime_powers:
        solutions_for_prime_power.append(_ge_solve_mod_n(aug_matrix, pp))
    
    solutions_for_variables = zip(*solutions_for_prime_power)
    
    for solutions_for_variable in solutions_for_variables:
        solutions.append(utils.chinese_remainder_theorem(list(
            zip(solutions_for_variable, prime_powers))))
    
    return solutions
  
solve = lu_method
  
def main ():
    # Important: array should have type = double.
    """
    test = np.array([
        [1., 1., 1., 3.],
        [2., 3., 7., 0.],
        [1., 3., -2., 17.]
    ])
    print(simple_ge(test))
    print(partial_pivoting_ge(test))
    print(partial_pivoting_with_scaling_ge(test))
    """
    
    A = np.array([
        [1., 2., 1., 4.],
        [2., 0., 4., 3.],
        [4., 2., 2., 1.],
        [-3., 1., 3., 2.]
    ])
    b = np.array([13., 28., 20., 6.])
    
    print(lu_factorise(A))
    print(lu_method(augment_matrix(A, b)))

if __name__ == '__main__':
    main()