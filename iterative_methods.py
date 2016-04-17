import numpy as np

from float_arithmetic import MACHINE_EPSILON
from math import sqrt

def _calculate_relative_error (old_guess, new_guess):
    return (np.linalg.norm(new_guess - guess, float("inf")) /
        (np.linalg.norm(new_guess, float("inf")) + MACHINE_EPSILON))

def _general_iterative_method (new_guess_init, guess_updater,
        aug_matrix, err_bound, iteration_bound, initial_guess):
    """
    initial_guess - numpy array.
    """
    if initial_guess is None:
        guess = np.zeros(len(aug_matrix))
    else:
        guess = initial_guess
    
    iterations = 0
    rel_err = err_bound + 1
    while rel_err >= err_bound and (
            iteration_bound is None or iterations < iteration_bound):
        new_guess = new_guess_init(aug_matrix, guess)
        
        for n in range(len(aug_matrix)):
            guess_updater(aug_matrix, n, guess, new_guess)
        
        rel_err = _calculate_relative_error(guess, new_guess)
        
        guess = new_guess
        
        iterations += 1
    
    return guess

def _guess_updater (aug_matrix, n, base_guess, new_guess):
    new_guess[n] = (1/aug_matrix[n, n]) * (aug_matrix[n, -1] -
        sum(aug_matrix[n, :n] * base_guess[:n]) -
        sum(aug_matrix[n, n+1:-1] * base_guess[n+1:]))
    
def _jacobi_new_guess_init (aug_matrix, old_guess):
    return np.zeros(len(aug_matrix))

def _jacobi_guess_updater (aug_matrix, n, old_guess, new_guess):
    _guess_updater(aug_matrix, n, old_guess, new_guess)
    
def jacobi_method (aug_matrix, err_bound, initial_guess=None,
        iteration_bound=None):
    return _general_iterative_method(_jacobi_new_guess_init,
        _jacobi_guess_updater, aug_matrix, err_bound, iteration_bound,
        initial_guess)

def _gauss_seidel_new_guess_init (aug_matrix, old_guess):
    return np.copy(old_guess)

def _gauss_seidel_guess_updater (aug_matrix, n, old_guess, new_guess):
    _guess_updater(aug_matrix, n, new_guess, new_guess)

def gauss_seidel_method (aug_matrix, err_bound, initial_guess=None
        iteration_bound=None):
    return _general_iterative_method(_gauss_seidel_new_guess_init,
        _gauss_seidel_guess_updater, aug_matrix, err_bound,
        iteration_bound, initial_guess)

"""
def _get_weighted_gauss_seidel_guess_updater (weight):
    def _weighted_gauss_seidel_guess_updater (aug_matrix, n, old_guess,
            new_guess):
        _gauss_seidel_guess_updater(aug_matrix, n, old_guess, new_guess)
        new_guess[n] = weight * new_guess[n] + (1.0 - weight) * old_guess[n]
    
    return _weighted_gauss_seidel_guess_updater

def _calculate_optimal_weight (
    optimal_weight = 2 / (1 + sqrt(1 - 
    
def weighted_gauss_seidel_method (aug_matrix, err_bound, initial_guess=None):
"""

def main ():
    test = np.array([
        [9., 1., 1., 10.],
        [2., 10., 3., 19.],
        [3., 4., 11., 0.]
    ])
    
    result = jacobi_method(test, 0.00001)
    for n in result:
        print(round(n, 4))
        
    result = gauss_seidel_method(test, 0.00001)
    for n in result:
        print(round(n, 4))

if __name__ == '__main__':
    main()