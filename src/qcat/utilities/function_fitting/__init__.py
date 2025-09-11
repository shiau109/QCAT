# QCAT function fitting package
"""
Usage Example:
-------------
from qcat.function_fitting import get_fitter
import xarray as xr

data = xr.DataArray(
	data=[1, 2, 3],
	dims=["x"],
	coords={"x": [0, 1, 2]}
)

fitter = get_fitter('cosine', data)
result = fitter.fit()
print(result.fit_report())
"""


# Import all fitter modules to ensure decorators run and registry is populated
from . import fit_cosine, fit_damped_oscillation, fit_exp_decay, fit_powerlaw_base, fit_transmon_freqeuency_flux
from .function_fitting import FunctionFitting, get_fitter

__all__ = [
	'FunctionFitting',
	'get_fitter',
]
