import numpy as np

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
            raise Exception("Failed to find pivot!")
        elif pivot_row != row:
            # Have to swap rows to get pivot in place.
            matrix[[row, pivot_row], :] = matrix[
                [pivot_row, row], :]
        
        # Now zero the values below the pivot in that column.
        for i in range(row + 1, len(matrix)):
            if matrix[i, row] != 0:
                matrix[i] -= (matrix[i, row] /
                    matrix[row, row]) * matrix[row]
        
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
            if pivot_row is None or (pivot_evaluator(matrix, i) >
                    best_pivot_val):
                pivot_row = i
                best_pivot_val = matrix[i, row]
        
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
    """Returns list of solutions to unknowns for UT form matrix."""
    solutions = []
    
    i = len(matrix) - 1
    while i >= 0:
        solution = (matrix[i, -1] -
            np.sum(matrix[i, i+1:-1] * solutions)) / matrix[i, i]
        solutions.insert(0, solution)
        
        i -= 1
        
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

def main ():
    # Important: array should have type = double.
    test = np.array([
        [1., 1., 1., 3.],
        [2., 3., 7., 0.],
        [1., 3., -2., 17.]
    ])
    print(simple_ge(test))
    print(partial_pivoting_ge(test))
    print(partial_pivoting_with_scaling_ge(test))

if __name__ == '__main__':
    main()