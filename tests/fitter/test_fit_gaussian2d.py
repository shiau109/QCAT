import numpy as np
from qcat.function_fitting.fit_gaussian2d import FitGaussian2D, FitMultiGaussian2D

def test_fit_gaussian2d():
    x = np.linspace(-5, 5, 40)
    y = np.linspace(-5, 5, 60)
    X, Y = np.meshgrid(x, y)
    amp, x0, y0, sigma_x, sigma_y, offset = 3.0, 1.0, -2.0, 1.5, 2.0, 0.5
    data = amp * np.exp(-(((X - x0) ** 2) / (2 * sigma_x ** 2) + ((Y - y0) ** 2) / (2 * sigma_y ** 2))) + offset
    fitter = FitGaussian2D(data, x, y)
    result = fitter.fit()
    params = result.best_values
    print(params)
    assert abs(params['amp'] - amp) < 0.5
    assert abs(params['x0'] - x0) < 0.5
    assert abs(params['y0'] - y0) < 0.5
    assert abs(params['sigma_x'] - sigma_x) < 0.5
    assert abs(params['sigma_y'] - sigma_y) < 0.5
    assert abs(params['offset'] - offset) < 0.5

def test_fit_multigaussian2d():
    x = np.linspace(-5, 5, 40)
    y = np.linspace(-5, 5, 60)
    X, Y = np.meshgrid(x, y)
    params_list = [
        (2.0, -2.0, -2.0, 1.0, 1.0),
        (1.5, 2.0, 2.0, 1.2, 1.5),
        (1.0, 0.0, 3.0, 0.8, 1.2)
    ]
    offset = 0.3
    data = np.zeros_like(X)
    for amp, x0, y0, sigma_x, sigma_y in params_list:
        data += amp * np.exp(-(((X - x0) ** 2) / (2 * sigma_x ** 2) + ((Y - y0) ** 2) / (2 * sigma_y ** 2)))
    data += offset
    fitter = FitMultiGaussian2D(data, x, y, n_gauss=3)
    result = fitter.fit()
    # No strict asserts, but print best values for inspection
    print("Best fit parameters:", result.best_values)
