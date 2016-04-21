import numpy as np

"""
import matplotlib.pyplot as plt
from math import sin, pi
"""

import lin_systems as ls
import iterative_methods as iter

def vandermonde (points):
    """
    points - [(x1, f(x1)), (x2, f(x2)), ...]
    """
    n = len(points)
    
    equations = []
    
    for x, fx in points:
        equation = []
        for e in range(n):
            equation.append(x ** e)
        
        equation.append(fx)
        equations.append(equation)
    
    return ls.partial_pivoting_with_scaling_ge(np.array(equations))

def vandermonde_reduced (points):
    mean = sum(x for x, _ in points) / len(points)
    reduced_points = [(x - mean, fx) for x, fx in points]
    return vandermonde(reduced_points)
    
def lagrange (x, points):
    result = 0
    for k, (_, fxk) in enumerate(points):
        result += fxk * lagrangian(k, x, points)
    
    return result

def lagrangian (k, x, points):
    xk = points[k][0]

    result = 1
    for i, (xi, _) in enumerate(points):
        if i != k:
            result *= (x - xi) / (xk - xi)
    
    return result
    
def evaluate_poly (poly, x):
    """
    `poly` = [c1, c2, ..., cn]
    Returns P(x) = c1 + c2*x + ... + cn*x^(n-1)
    """
    return sum(c * (x ** n) for n, c in enumerate(poly))

def multiply_polys (p1, p2):
    new_degree = (len(p1) - 1) + (len(p2) - 1)
    new_poly = [0 for _ in range(new_degree + 1)]
    
    for deg1, c1 in enumerate(p1):
        for deg2, c2 in enumerate(p2):
            new_poly[deg1 + deg2] += c1 * c2
    
    return new_poly

def add_polys (p1, p2):
    smaller_poly, bigger_poly = (p1, p2) if len(p1) < len(p2) else (p2, p1)

    new_poly = bigger_poly[:]
    for i, c in enumerate(smaller_poly):
        new_poly[i] += c
    
    return new_poly
    
def multiply_poly_by_scalar (scalar, poly):
    return [scalar * c for c in poly]

def differentiate_poly (poly):
    return [i * poly[i] for i in range(1, len(poly))]

def poly_to_func (poly):
    def f (x):
        return evaluate_poly(poly, x)
    return f
    
def get_w_poly (points):
    poly = [1]

    factors = [[-x, 1] for x, _ in points]
    for factor in factors:
        poly = multiply_polys(poly, factor)
    
    return poly
    
def newton_add_point (point, points, poly):
    x, fx = point
    if len(points) == 0:
        return [fx]
    
    w = get_w_poly(points)
    a = (fx - evaluate_poly(poly, x)) / evaluate_poly(w, x)
    
    return add_polys(poly, multiply_poly_by_scalar(a, w))
    
def newton_basis (points):
    processed_points = []
    poly = []
    for point in points:
        poly = newton_add_point(point, processed_points, poly)
        processed_points.append(point)
    
    return poly

def derivative_of_unknown_fn (points):
    return poly_to_func(differentiate_poly(newton_basis(points)))
    
def get_local_extrema (points, poly, err_bound):
    diff_poly = differentiate_poly(poly)
    
    _f = poly_to_func(diff_poly)
    def f (x):
        return np.array([_f(x)])
    _J = poly_to_func(differentiate_poly(diff_poly))
    def J (x):
        return np.array([_J(x)])
    
    local_extrema = []
    for x, _ in points:
        try:
            extremum = iter.newton_raphson(f, J, err_bound, np.array([x]))[0]
            local_extrema.append((extremum, evaluate_poly(poly, extremum)))
        except ls.PivotNotFoundException:
            # Sometimes fails to calculate dx?
            pass

    return local_extrema

def _get_global_extremum (extremum_getter, points, poly, err_bound):
    return extremum_getter(get_local_extrema(points, poly, err_bound),
        key=lambda x: x[1])

def get_global_max (points, poly, err_bound):
    return _get_global_extremum(max, points, poly, err_bound)

def get_global_min (points, poly, err_bound):
    return _get_global_extremum(min, points, poly, err_bound)

def main ():
    # Testing extrema stuff.
    pts = [(2, 312), (4, 330), (6, 252), (7, 192), (8, 126), (9, 60)]
    poly = newton_basis(pts)
    
    print(get_global_max(pts, poly, 0.000001))

    # Testing Newton.
    """
    points = [(0, 0), (pi/2, 1), (pi, 0)]
    
    poly = newton_basis(points)
    #F = poly_to_func(poly)
    F = poly_to_func(newton_add_point((-3, sin(-3)), points, poly))
    
    xs = []
    fxs = []
    sins = []
    
    x = -5
    while x < 5:
        xs.append(x)
        fxs.append(F(x))
        sins.append(sin(x))
    
        x += 0.1
        
    plt.plot(xs, fxs)
    plt.plot(xs, sins)
    plt.show()
    """
    
    # Testing Lagrange.
    """
    lagrange_pts = [(0.5, -0.693), (1.0, 0.0), (2.0, 0.693)]

    xs = []
    fxs = []
    logs = []
    
    x = 0.1
    while x < 5.0:
        xs.append(x)
        fxs.append(lagrange(x, lagrange_pts))
        logs.append(log(x))
        
        x += 0.1
    
    plt.plot(xs, fxs)
    plt.plot(xs, logs)
    plt.show()
    """

if __name__ == '__main__':
    main()