import os
import xarray as xr


def load_xarray_h5(file_path: str, engine_order: list[str] | None = None, load_into_memory: bool = True) -> "xr.Dataset":
    """
    Load an xarray.Dataset stored in an HDF5 (.h5) file.

    Parameters
    ----------
    file_path : str
        Path to the .h5 file.
    group : str | None
        HDF5 group name where the dataset is stored (if any).
    engine_order : list[str] | None
        List of xarray engines to try (default: ["h5netcdf", "netcdf4"]).
    load_into_memory : bool
        If True, call .load() on the returned dataset to read it into memory.

    Returns
    -------
    xr.Dataset
        The loaded xarray Dataset.

    Raises
    ------
    FileNotFoundError
        If file_path does not exist.
    RuntimeError
        If the file cannot be opened as an xarray Dataset with the tried engines.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    engines = engine_order or ["h5netcdf", "netcdf4"]
    last_exc = None
    for eng in engines:
        try:
            ds = xr.open_dataset(file_path, engine=eng)
            if load_into_memory:
                ds = ds.load()
            return ds
        except Exception as exc:
            last_exc = exc

    # Final fallback: let xarray choose the engine
    try:
        ds = xr.open_dataset(file_path)
        if load_into_memory:
            ds = ds.load()
        return ds
    except Exception as exc:
        raise RuntimeError(f"Failed to open '{file_path}' as an xarray Dataset. Tried engines {engines}. Last error: {last_exc}") from exc

def repetition_data( ds: xr.Dataset, repetition_dim: str = "qubit"):
    n_qubits = ds.sizes[repetition_dim]
    output_data = []
    for qubit_idx in range(n_qubits):
        data = ds.isel(**{repetition_dim: qubit_idx})
        output_data.append(data)
    return output_data


import datetime

def parse_timestamp(ts):
    # Remove timezone info for parsing
    if '+' in ts:
        ts = ts.split('+')[0]
    return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")

# def to_NCU_Measurement_Master(ds: xr.Dataset, repetition_dim: str = "qubit"):
#     from types import SimpleNamespace
#     from qcat.NCU.Measurement_Master import Measurement_Save
#     Measurement_Save()
#     Raw_data = SimpleNamespace(
#             first_samples=ds.coords["detuning"].values,
#             second_samples=ds.coords["readout_amp_ratio"].values
#     )


if __name__ == "__main__":
    ds = load_xarray_h5(r"d:\github\ASQMDriver\data\MIST\2025-09-08\#68_07_iq_blobs_210029\ds_raw.h5")
    # print(ds)
    sep_data = repetition_data(ds, repetition_dim="qubit")
    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        print(qubit_name, type(qubit_name))