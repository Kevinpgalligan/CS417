import numpy as np

import iterative_methods as iter

from math import pi, sin, cos, tan

def main ():
    def f (q, v, t):
        return np.array([
            v * cos(q) * t - 300,
            (-0.5)*9.81*(t ** 2) + v*sin(q)*t - 71,
            ((-9.81)*t + v*sin(q))/(v*cos(q)) + 1
        ])
    
    def J (q, v, t):
        return np.array([
            [(-v)*t*sin(q), t*cos(q), v*cos(q)],
            [v*t*cos(q), t*sin(q), (-9.81)*t + v*sin(q)],
            [tan(q) * ((-9.81)*t/(v*cos(q))) + (1/cos(q))**2,
                (-9.81*t)/((v**2)*cos(q)),
                (-9.81)/(v*cos(q))]
        ])
    
    init_guess = np.array([61.423158, 0.974487, 8.696960])
    print(f(*init_guess))
    ans = iter.newton_raphson(f, J, 0.0000001, init_guess)
    print(ans)
    print(f(*ans))

if __name__ == '__main__':
    main()