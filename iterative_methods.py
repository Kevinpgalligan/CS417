import numpy as np

from float_arithmetic import MACHINE_EPSILON
import lin_systems as ls

from math import sqrt

def _calculate_relative_error (old_guess, new_guess):
    return (np.linalg.norm(new_guess - old_guess, float("inf")) /
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
    
    return guess, rel_err, iterations

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
        initial_guess)[0]

def _gauss_seidel_new_guess_init (aug_matrix, old_guess):
    return np.copy(old_guess)

def _gauss_seidel_guess_updater (aug_matrix, n, old_guess, new_guess):
    _guess_updater(aug_matrix, n, new_guess, new_guess)

def gauss_seidel_method (aug_matrix, err_bound, initial_guess=None,
        iteration_bound=None):
    return _general_iterative_method(_gauss_seidel_new_guess_init,
        _gauss_seidel_guess_updater, aug_matrix, err_bound,
        iteration_bound, initial_guess)[0]

def _get_weighted_gauss_seidel_guess_updater (weight):
    def _weighted_gauss_seidel_guess_updater (aug_matrix, n, old_guess,
            new_guess):
        _gauss_seidel_guess_updater(aug_matrix, n, old_guess, new_guess)
        new_guess[n] = weight * new_guess[n] + (1.0 - weight) * old_guess[n]
    
    return _weighted_gauss_seidel_guess_updater

def _calculate_optimal_weight (dx, dxp1):
    optimal_weight = 2 / (1 + sqrt(1 - (dxp1 / dx) ** (1 / p)))
    return optimal_weight
    
def weighted_gauss_seidel_method (aug_matrix, err_bound, initial_guess=None):
    guess_updater = _get_weighted_gauss_seidel_guess_updater(1.0)

    x9, err, _ = _general_iterative_method(_gauss_seidel_new_guess_init,
        guess_updater, aug_matrix, err_bound, 9, initial_guess)
    if err < err_bound:
        return x9
    
    x10, err, _ = _general_iterative_method(_gauss_seidel_new_guess_init,
        guess_updater, aug_matrix, err_bound, 1, x9)
    if err < err_bound:
        return x10
    
    x11, err, _ = _general_iterative_method(_gauss_seidel_new_guess_init,
        guess_updater, aug_matrix, err_bound, 1, x9)
    if err < err_bound:
        return x11
    
    dx10 = np.linalg.norm(x9 - x10)
    dx11 = np.linalg.norm(x10 - x11)
    optimal_weight = _calculate_optimal_weight(dx10, dx11)
    
    guess_updater = _get_weighted_gauss_seidel_guess_updater(optimal_weight)
    
    return _general_iterative_method(_gauss_seidel_new_guess_init,
        guess_updater, aug_matrix, err_bound, None, x11)[0]

def newton_raphson (f, J, err_bound, initial_guess=None):
    if initial_guess is None:
        x = np.zeros(f.__code__.co_argcount)
    else:
        x = initial_guess
    
    iterations = 0
    
    fx = f(*x)
    err = err_bound + 1
    while err >= err_bound:
        try:
            dx = ls.solve(ls.augment_matrix(J(*x), -fx))
        except ls.PivotNotFoundException:
            print(iterations)
            return x
        
        x = x + dx
        new_fx = f(*x)
        err = np.linalg.norm(fx - new_fx)
        fx = new_fx
        
        iterations += 1
    
    return x
    
def main ():
    """Iterative methods for linear systems.
    test = np.array([
        [9., 1., 1., 10.],
        [2., 10., 3., 19.],
        [3., 4., 11., 0.]
    ])
    
    for method in [jacobi_method, gauss_seidel_method,
            weighted_gauss_seidel_method]:
        result = method(test, 0.00000001)
        for n in result:
            print(round(n, 6))
    """
    
    def f (x1, x2):
        return np.array([
            1.4 * x1 - x2 - 0.6,
            (x1 ** 2) - 1.6 * x1 - x2 - 4.6
        ])
    
    def J (x1, x2):
        return np.array([
            [1.4, -1],
            [2 * x1 - 1.6, -1]
        ])
    
    print(newton_raphson(f, J, 0.00001))
        
if __name__ == '__main__':
    main()