import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


def differentiation():
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    dy_dx = np.gradient(y, x)

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label="sin(x)")
    plt.plot(x, dy_dx, label="d(sin(x))/dx")
    plt.legend()
    plt.title("Differentiation of sin(x)")
    if not os.path.exists("./assets"):
        os.makedirs("./assets")
    print("saving the differentian to file..")
    plt.savefig("./assets/differentiation.png")


def numerical-integration():
    def f(x):
        return x**2

    result, error = integrate.quad(f, 0, 1)
    print(f"Integral of x^2 from 0 to 1: {result}")


def curve-fitting():
    x = np.linspace(0, 10, 100)
    y = 3 * x**2 + 2 * x + 1 + np.random.normal(0, 10, 100)

    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)

    plt.figure(figsize=(10, 4))
    plt.scatter(x, y, label="Data")
    plt.plot(x, poly(x), color="r", label="Fitted Curve")
    plt.legend()
    plt.title("Curve Fitting")
    if not os.path.exists("./assets"):
        os.makedirs("./assets")
    print("saving the curve_fitting to file..")
    plt.savefig("./assets/curve_fitting.png")


def linear-regression():
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2, 4, 5, 4, 5])

    model = LinearRegression()
    model.fit(x, y)

    plt.figure(figsize=(10, 4))
    plt.scatter(x, y, label="Data")
    plt.plot(x, model.predict(x), color="r", label="Regression Line")
    plt.legend()
    plt.title("Linear Regression")
    plt.savefig("./assets/linear_regression.png")


def spline-interpolation():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 8, 9, 1, -7, 6])

    f = interp1d(x, y, kind="cubic")
    x_new = np.linspace(0, 5, 100)
    y_new = f(x_new)

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, "o", label="Data points")
    plt.plot(x_new, y_new, label="Cubic Spline")
    plt.legend()
    plt.title("Spline Interpolation")
    if not os.path.exists("./assets"):
        os.makedirs("./assets")
    print("saving the spline_interpolation to file..")
    plt.savefig("./assets/spline_interpolation.png")


differentiation()
numerical-integration()
curve-fitting()
linear-regression()
spline-interpolation()
