import numpy as np


def trapezoidal-rule(f, a, b, n):
    """
    Approximate the definite integral of f from a to b using the trapezoidal rule.

    Parameters:
    f -- function to integrate
    a, b -- interval of integration [a,b]
    n -- number of subintervals
    """
    x = np.linspace(a, b, n + 1)
    y = f(x)

    return (b - a) / (2 * n) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])


def f(x):
    return x**2


a, b = 0, 1  # Interval of integration
n = 1000  # Number of subintervals

result = trapezoidal-rule(f, a, b, n)
print(f"Approximate integral of x^2 from 0 to 1: {result}")
print(f"Exact value: {1/3}")
print(f"Error: {abs(result - 1/3)}")
