import xarray as xr
import numpy as np
from qcat.function_fitting import get_fitter

def test_exp_decay():
    x = np.linspace(0, 10, 100)
    y = 3 * np.exp(-x/2) + 0.5
    data = xr.DataArray(y, dims=["x"], coords={"x": x})
    fitter = get_fitter('exp_decay', data)
    result = fitter.fit()
    params = result.best_values
    assert abs(params['a'] - 3) < 0.3
    assert abs(params['tau'] - 2) < 0.3
    assert abs(params['c'] - 0.5) < 0.2
