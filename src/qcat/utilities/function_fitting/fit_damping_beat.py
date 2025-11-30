from xarray import DataArray
from lmfit import Model, Parameter
from lmfit.model import ModelResult
from numpy import ndarray, fft, linspace, asarray
from numpy import cos, abs, exp, max, min, mean, argmax
from numpy import pi, nan
from .function_fitting import FunctionFitting, register_fitter

@register_fitter('damping_beat')
class FitDampingBeat(FunctionFitting):
    """
    Fit a damped beat model to data:
    a * exp(-x/tau) * (cos(2*pi*f_1*x + phi_1) + cos(2*pi*f_2*x + phi_2)) + c
    """
    def __init__(self, data:DataArray=None):
        self._data_parser(data)
        self.model = Model(self.model_function)
        self.params = None

    def _data_parser(self, data:DataArray):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")
        self.y = data.values
        self.x = data.coords["x"].values

    def model_function(self, x, a_1, kappa_1, f_1, phi_1, a_2, kappa_2, f_2, phi_2, c):
        return  a_1 *exp(-x*kappa_1) *cos(2*pi*f_1*x + phi_1) + a_2*exp(-x*kappa_2)*cos(2*pi*f_2*x + phi_2) + c

    def guess(self):
        y = self.y
        t = self.x
        dt = float(t[1] - t[0])
        max_val = float(max(y))
        min_val = float(min(y))
        # FFT for frequency guesses
        amp = fft.fft(y)[: len(y) // 2]
        freq = fft.fftfreq(len(y), dt)[: len(amp)]
        amp[0] = 0  # Remove DC part
        power = abs(amp)
        peak_indices = asarray(power).argsort()[::-1]
        freq = asarray(freq)
        f_1_idx = peak_indices[0]
        # Find second peak index with sufficient separation
        f_2_idx = None
        for idx in peak_indices[1:5]:
            if abs(idx - f_1_idx) >= 3 and power[idx]/power[f_1_idx]>0.5:
                f_2_idx = idx
                break
        f_1_guess = float(abs(freq[f_1_idx]))
        a_1_guess = float(abs(amp[f_1_idx]))
        a_1_guess_dict = dict(value=a_1_guess, min=0.0, max=a_1_guess*2)
        f_1_guess_dict = dict(value=f_1_guess, min=0.0, max=1.0/dt/2)
        phi_1_guess_dict = dict(value=0.0, min=-float(pi), max=float(pi))
        # kappa_1 guess: use 1/(t[-1]/2) as typical decay rate
        kappa_1_guess = 1.0 / abs(t[-1]/2) if abs(t[-1]/2) > 0 else 1.0
        kappa_1_guess_dict = dict(value=kappa_1_guess, min=0, max=10*kappa_1_guess)
        # If second frequency is not resolvable, fit single frequency only
        if f_2_idx is None:
            a_2_guess_dict = dict(value=0, vary=False)
            f_2_guess_dict = dict(value=0, vary=False)
            phi_2_guess_dict = dict(value=0, min=-float(pi), max=float(pi), vary=False)
            kappa_2_guess_dict = dict(value=0, min=0, max=kappa_1_guess, vary=False)
        else:
            f_2_guess = float(abs(freq[f_2_idx]))
            a_2_guess = float(abs(amp[f_2_idx]))
            a_2_guess_dict = dict(value=a_2_guess, min=0.0, max=a_2_guess*2)
            f_2_guess_dict = dict(value=f_2_guess, min=0.0, max=1.0/dt/2)
            phi_2_guess_dict = dict(value=0.0, min=-float(pi), max=float(pi))
            # kappa_2 guess: same as kappa_1
            kappa_2_guess_dict = dict(value=kappa_1_guess, min=0, max=10*kappa_1_guess)
        c_guess_dict = dict(value=float(mean(y)), min=min_val, max=max_val)
        self.params = self.model.make_params(
            a_1=a_1_guess_dict,
            kappa_1=kappa_1_guess_dict,
            a_2=a_2_guess_dict,
            kappa_2=kappa_2_guess_dict,
            f_1=f_1_guess_dict,
            phi_1=phi_1_guess_dict,
            f_2=f_2_guess_dict,
            phi_2=phi_2_guess_dict,
            c=c_guess_dict
        )
        return self.params

    def fit(self, data:DataArray=None) -> ModelResult:
        if data is not None:
            self._data_parser(data)
        if self.params is None:
            self.guess()
        result = self.model.fit(self.y, self.params, x=self.x)
        self.result = result
        return result
