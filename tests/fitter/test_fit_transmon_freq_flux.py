import xarray as xr
import numpy as np
from qcat.function_fitting import get_fitter

def test_transmon_freq_flux():
    x = np.linspace(-0.2, 0.2, 10)
    Ec = 0.2
    Ej_sum = 10
    offset = 0.0
    period = 0.6
    d = 0.0
    quan_flux = (x - offset) / period
    Ej_eff = Ej_sum * np.abs(np.cos(np.pi * quan_flux))
    y = np.sqrt(8 * Ec * Ej_eff) - Ec
    data = xr.DataArray(y, dims=["x"], coords={"x": x})
    fitter = get_fitter('transmon_freq_flux', data)
    result = fitter.fit()
    params = result.best_values
    assert abs(params['Ec'] - Ec) < 0.1
    assert abs(params['Ej_sum'] - Ej_sum) < 2
