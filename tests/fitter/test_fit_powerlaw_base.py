import xarray as xr
import numpy as np
from qcat.function_fitting import get_fitter

def test_powerlaw_base():
    x = np.linspace(0, 5, 100)
    y = 2 * (0.8 ** x) + 0.3
    data = xr.DataArray(y, dims=["x"], coords={"x": x})
    fitter = get_fitter('powerlaw_base', data)
    result = fitter.fit()
    params = result.best_values
    assert abs(params['a'] - 2) < 0.3
    assert abs(params['base'] - 0.8) < 0.1
    assert abs(params['c'] - 0.3) < 0.2
