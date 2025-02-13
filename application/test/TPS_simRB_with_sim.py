import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from qcat.analysis.qubit.clifford_1QRB import Clifford1QRB

############################################
# Section 1: Plot individual STD vs. Gate Number curves
############################################

# Define the transformed coupler frequency array (to be used as the y-axis in the colormap)
# (Note: This array has 17 points. Make sure that the number of measurements matches this.)
x_array = np.array([-0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 
                    0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
coupler_freqs = np.sqrt(8 * 0.2 * 27 * np.abs(np.cos((x_array + 0.115) / 0.7 * np.pi))) - 0.2

# Folder list and other parameters
folder_list = ["independent", "simultaneous"]
ro_name = "q1_ro"
root_dir = Path(r"D:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\1QRB_2")

# Loop over each folder to plot STD vs. gate number curves.
for folder_name_str in folder_list:
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Gate Number", fontsize=20)
    ax.set_ylabel("STD", fontsize=20)
    ax.set_xscale("log")
    
    folder_path = root_dir / folder_name_str
    # Get all subfolders (each expected to contain a file "1QRB.nc")
    folder_names = [item.name for item in folder_path.iterdir() if item.is_dir()]
    print(f"Folder '{folder_name_str}' has {len(folder_names)} subfolders.")
    
    for subfolder_name in folder_names:
        file_path = folder_path / subfolder_name / "1QRB.nc"
        dataset = xr.open_dataset(str(file_path))
        sq_data = dataset[ro_name]
        # Keep attributes and name as before
        sq_data.attrs = dataset.attrs
        sq_data.name = ro_name
        
        # Build a new Dataset with coordinate "gate_num" taken from the file's x axis.
        data = xr.Dataset(
            {
                "p0": (("gate_num",), sq_data.sel(mixer="val").values),
                "std": (("gate_num",), sq_data.sel(mixer="err").values),
            },
            coords={"gate_num": sq_data.coords["x"].values},
        )
        label = sq_data.name
        
        # Optionally run the analysis (for any side effects)
        my_ana = Clifford1QRB(data, label)
        my_ana._start_analysis(plot=False)
        
        # Plot STD vs. gate number for this measurement.
        ax.plot(data.coords["gate_num"].values, data["std"].values, "-", label=subfolder_name)
    
    plt.tight_layout()
    plt.show()

############################################
# Section 2: Build and plot 2D colormaps using pcolormesh
############################################

# For the 2D colormap, we collect STD data from each measurement file into a 2D array.
# Each row will correspond to one measurement file.
colormap_data = {}

for folder_name_str in folder_list:
    folder_path = root_dir / folder_name_str
    # Sort the subfolders to enforce an order. (Make sure that the order corresponds to the coupler_freqs.)
    folder_names = sorted([item.name for item in folder_path.iterdir() if item.is_dir()])
    std_list = []
    gate_num_common = None  # We'll assume all files share the same gate number array.
    print(f"Collecting colormap data for folder '{folder_name_str}' with {len(folder_names)} subfolders...")
    
    for subfolder_name in folder_names:
        file_path = folder_path / subfolder_name / "1QRB.nc"
        dataset = xr.open_dataset(str(file_path))
        sq_data = dataset[ro_name]
        gate_num = sq_data.coords["x"].values  # Use these values as the x-axis centers.
        if gate_num_common is None:
            gate_num_common = gate_num
        data = xr.Dataset(
            {
                "p0": (("gate_num",), sq_data.sel(mixer="val").values),
                "std": (("gate_num",), sq_data.sel(mixer="err").values),
            },
            coords={"gate_num": gate_num},
        )
        std_list.append(data["std"].values)
    
    # Convert list to a 2D array: rows = measurements (ordered as in folder_names)
    std_2d = np.array(std_list)
    colormap_data[folder_name_str] = (gate_num_common, std_2d)

# Since gate_num is not uniformly spaced, we compute the bin edges from the centers.
def compute_edges(centers):
    centers = np.array(centers)
    edges = np.empty(len(centers) + 1)
    # Calculate midpoints between consecutive centers.
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    # For the boundaries, use the first and last interval sizes.
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return edges

# Similarly, compute bin edges for the coupler frequency axis.
# (We assume the number of measurements equals len(coupler_freqs).)
if 'expected_num' not in locals():
    expected_num = len(coupler_freqs)  # expected number of measurement files

############################################
# Option 1: Plot each folder’s colormap in its own figure using pcolormesh
############################################
for folder_name_str, (gate_num, std_2d) in colormap_data.items():
    # Compute x-axis edges (gate number) from nonuniform centers.
    x_edges = compute_edges(gate_num)
    
    # Check if the number of measurements matches the number of coupler_freqs.
    if std_2d.shape[0] != len(coupler_freqs):
        print(f"Warning: For folder '{folder_name_str}', number of measurements ({std_2d.shape[0]}) "
              f"differs from number of coupler frequency points ({len(coupler_freqs)}).")
        # In this case, fall back to measurement indices.
        y_centers = np.arange(std_2d.shape[0])
    else:
        y_centers = coupler_freqs

    y_edges = compute_edges(y_centers)
    
    fig, ax = plt.subplots(1, figsize=(8, 6))
    # Use pcolormesh to plot the 2D colormap on the nonuniform grid.
    mesh = ax.pcolormesh(x_edges, y_edges, std_2d, shading='auto', cmap='viridis')
    ax.set_xscale("log")
    ax.set_xlabel("Gate Number", fontsize=14)
    ax.set_ylabel("Coupler Frequency", fontsize=14)
    ax.set_title(f"2D Colormap: {folder_name_str}", fontsize=16)
    fig.colorbar(mesh, ax=ax, label="STD")
    plt.tight_layout()
    plt.show()
