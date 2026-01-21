import matplotlib.pyplot as plt
import xarray as xr

def plot_2d_colormap_from_h5(sq_data:xr.Dataset, data_var, x_dim, y_dim):
	
    qubit_name = str(sq_data["qubit"].values.item())

    z = sq_data[data_var]
    # If z is 1D, skip
    if z.ndim < 2:
        print(f"Skipping qubit {qubit_name}: data is not 2D.")
        return None
    fig, ax = plt.subplots(figsize=(8,6))
    sq_data[data_var].plot(ax=ax, x=x_dim, y=y_dim, add_colorbar=True, cmap="viridis")
    ax.set_xlabel(x_dim)
    ax.set_ylabel(y_dim)
    ax.set_title(f'Qubit {qubit_name}: {data_var} vs {x_dim}, {y_dim}')
    # fig.colorbar(c, ax=ax, label=data_var)
    fig.tight_layout()
    return fig