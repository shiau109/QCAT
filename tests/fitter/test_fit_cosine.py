import xarray as xr
import numpy as np
from qcat.function_fitting import get_fitter

def test_cosine():
    x = np.linspace(0, 2 * np.pi, 100)
    y = 2 * np.cos(2 * np.pi * 0.5 * x + 0.1) + 1.0
    data = xr.DataArray(y, dims=["x"], coords={"x": x})
    fitter = get_fitter('cosine', data)
    result = fitter.fit()
    params = result.best_values
    assert abs(params['a'] - 2) < 0.2
    assert abs(params['f'] - 0.5) < 0.05
    assert abs(params['c'] - 1.0) < 0.2
