from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
from numpy import exp, linspace, max, min, mean

from .function_fitting import FunctionFitting


class FitExponentialDecay(FunctionFitting):
    """
    Class for fitting exponential decay data.
    """
    def __init__(self, data: DataArray = None):
        self._data_parser(data)
        self.model = Model(self.model_function)

    def _data_parser(self, data: DataArray):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")

        self.y = data.values
        self.x = data.coords["x"].values

    def model_function(self, x, a, tau, c):
        """
        Exponential decay model: y = a * exp(-x / tau) + c
        """
        return a * exp(-x / tau) + c

    def guess(self):
        """
        Generate initial parameter guesses for the model.
        """
        y = self.y
        x = self.x

        # Amplitude guess
        a_guess = y[0] - y[-1]
        if a_guess < 0:
            a_guess_dict = dict(value=a_guess, min=a_guess * 2, max=0)
        else:
            a_guess_dict = dict(value=a_guess, min=0, max=a_guess * 2)
        

        # Decay constant guess (tau)
        tau_guess_dict = dict(value=x[-1] / 2, min=0, max=x[-1] * 2)

        # Offset guess (c)
        c_guess_dict = dict(value=y[-1], min=y[-1] -abs(a_guess), max=y[-1] +abs(a_guess))

        params = self.model.make_params(
            a=a_guess_dict, tau=tau_guess_dict, c=c_guess_dict
        )

        self.params = params
        return params

    def fit(self, data: DataArray = None) -> ModelResult:
        """
        Perform the fitting process on the provided data.
        """
        if data is not None:
            self._data_parser(data)

        self.guess()

        result = self.model.fit(self.y, self.params, x=self.x)
        self.result = result
        return result
