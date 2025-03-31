from xarray import DataArray
from lmfit import Model, Parameter
from lmfit.model import ModelResult
import numpy as np
import matplotlib.pyplot as plt
from qcat.analysis.function_fitting.function_fitting import FunctionFitting

class FitTransmonFrequencyFlux(FunctionFitting):
    """
    Fit transmon frequency vs. flux data.
    
    The model function is defined as:
    
        f(x) = sqrt(8 * Ec * Ej_eff) - Ec
    
    with:
    
        Ej_eff = Ej_sum * sqrt( cos(π * quan_flux)**2 + (d * sin(π * quan_flux))**2 )
        quan_flux = (x - offset) / period
    
    In our case we force:
        - Ec = 0.2
        - d = 0  (so that Ej_eff = Ej_sum * |cos(π*quan_flux)|)
        - period = π  (so that π*quan_flux = x - offset)
        - offset is fixed at x₀ = -0.015 (to match the symmetry of the two equal points)
    
    Then the model reduces to:
    
        f(x) = sqrt(8*0.2*Ej_sum*|cos(x - x₀)|) - 0.2
    """
    def __init__(self, data: DataArray = None):
        self.Ec_design = 0.2
        self._data_parser(data)
        self.params = None
        self.model = Model(self.model_function)

    def _data_parser(self, inputdata: DataArray):
        if not isinstance(inputdata, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")
        self.freq = inputdata.values
        self.x = inputdata.coords["x"]

    def model_function(self, x, Ec, offset, period, Ej_sum, d):
        quan_flux = self.iv2quantFlux(x, offset, period)
        Ej_eff = self.effective_Ej(quan_flux, Ej_sum, d)
        return self.tramsmon_frequency(Ej_eff, Ec)

    def tramsmon_frequency(self, Ej, Ec):
        return np.sqrt(8 * Ec * Ej) - Ec

    def effective_Ej(self, quan_flux, Ej_sum, d):
        # When d=0, this becomes Ej_sum * |cos(π*quan_flux)|
        return Ej_sum * np.sqrt(np.cos(np.pi * quan_flux) ** 2 + (d * np.sin(np.pi * quan_flux)) ** 2)

    def iv2quantFlux(self, iv, offset, period):
        return (iv - offset) / period

    def guess(self):
        # For a very small dataset the automatic guess may not be ideal.
        # We therefore supply some reasonable starting guesses.
        y = self.freq
        x = self.x

        period_guess = 0.6
        # Set fixed parameters:
        Ec_guess_dict = dict(value=self.Ec_design, vary=False)
        period_guess_dict = dict(value=period_guess, min=0)
        d_guess_dict = dict(value=0, vary=False)
        offset_guess_dict = dict(value=0, max=period_guess/2, min=-period_guess/2)  # force symmetry

        # For Ej_sum, we make an initial guess based on one of the high points.
        # Using f(x) = sqrt(8*Ec*Ej_sum*|cos(x-x0)|)-Ec, at x where cos(x-x0) ~ cos(0.16) ~ 0.987,
        # and f(x) ~ 6.06, we have:
        #   sqrt(1.6*Ej_sum*0.987) = 6.06 + 0.2 = 6.26  => Ej_sum ~ (6.26^2)/(1.6*0.987)
        Ej_sum_guess = max(self.freq**2/self.Ec_design/8.)
        Ej_sum_guess_dict = dict(value=Ej_sum_guess, min=0)

        params = self.model.make_params(
            Ec=Ec_guess_dict,
            offset=offset_guess_dict,
            period=period_guess_dict,
            Ej_sum=Ej_sum_guess_dict,
            d=d_guess_dict,
        )
        self.params = params
        return params

    def fit(self, data: DataArray = None) -> ModelResult:
        if data is not None:
            self._data_parser(data)
        if self.params is None:
            self.guess()
        result = self.model.fit(self.freq, self.params, x=self.x)
        self.result = result
        return result

    def plot_fit(self, num_points=200):
        """
        Plot the data points along with the fitted curve.
        """
        if not hasattr(self, "result"):
            raise RuntimeError("No fit result available. Please run fit() first.")
            
        # Generate a smooth x-axis for the fitted curve
        x_fit = np.linspace(min(self.x), max(self.x), num_points)
        # Evaluate the model using the best-fit parameters
        best_values = self.result.params.valuesdict()
        y_fit = self.model_function(x_fit, **best_values)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_fit, y_fit, label="Fitted curve", color='blue')
        ax.scatter(self.x, self.freq, color='red', label="Data points", zorder=5)
        ax.set_xlabel("x")
        ax.set_ylabel("Frequency")
        ax.set_title("Transmon Frequency vs. Flux Fit")
        ax.legend()
        plt.show()


if __name__ == '__main__':
    import xarray as xr

    # Create a DataArray for the three data points:
    # Points: (-0.175, 6.06), (0.145, 6.06), (0.21, 4.725)
    data = xr.DataArray(
        data=np.array([6.06, 6.06, 4.730, 4.913]),#5.8]),
        dims=["x"],
        coords={"x": np.array([-0.170, 0.143, 0.208, 0.202])},#0.111])},
        name="transmon_freq",
        attrs={"description": "Three points for transmon frequency vs. flux"}
    )

    my_fit = FitTransmonFrequencyFlux(data)
    params = my_fit.guess()
    # Fix parameters to force the desired model:
    params["Ec"].set(value=0.16, vary=False)
    params["period"].set(value=0.8, vary=True)
    params["d"].set(value=0, vary=False)
    # params["offset"].set(value=-0.015, vary=False)  # symmetry axis from the two equal points

    # Perform the fit:
    result = my_fit.fit()
    print(result.fit_report())

    # Plot the data and the fitting curve:
    my_fit.plot_fit()
