import xarray as xr
import numpy as np
from qcat.function_fitting import get_fitter

def test_damped_oscillation():
    x = np.linspace(0, 10, 100)
    y = 2 * np.exp(-x/3) * np.cos(2 * np.pi * 0.5 * x + 0.1) + 1.0
    data = xr.DataArray(y, dims=["x"], coords={"x": x})
    fitter = get_fitter('damped_oscillation', data)
    result = fitter.fit()
    params = result.best_values
    assert abs(params['a'] - 2) < 0.3
    assert abs(params['tau'] - 3) < 0.5
    assert abs(params['f'] - 0.5) < 0.05
    assert abs(params['c'] - 1.0) < 0.3
