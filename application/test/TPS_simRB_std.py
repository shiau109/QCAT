import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from qcat.analysis.qubit.clifford_1QRB import Clifford1QRB

# List of folders to process
folder_list = ["independent", "simultaneous"]
ro_name = "q1_ro"

##############################
# Section 1: Scatter-Line Plots
##############################
# (This is your original section for plotting STD vs Gate Number curves)
for folder_name_str in folder_list:
    # Create a new figure for each folder
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Gate Number", fontsize=20)
    ax.set_ylabel("STD", fontsize=20)
    # ax.set_xscale("log")
    
    # Specify the folder path
    folder_path = Path(r"D:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\1QRB_2") / folder_name_str
    # Get all subfolder names (each assumed to contain a file "1QRB.nc")
    folder_names = [item.name for item in folder_path.iterdir() if item.is_dir()]
    print(f"Processing folder '{folder_name_str}' with {len(folder_names)} subfolders...")
    
    for subfolder_name in folder_names:
        # Build the full path to the data file
        file_path = folder_path / subfolder_name / "1QRB.nc"
        dataset = xr.open_dataset(str(file_path))
        
        # Get the raw data array from the dataset and set its attributes
        sq_data = dataset[ro_name]
        sq_data.attrs = dataset.attrs
        sq_data.name = ro_name

        # Reformat the data into a new Dataset with a coordinate "gate_num"
        data = xr.Dataset(
            {
                "p0": (("gate_num",), sq_data.sel(mixer="val").values),
                "std": (("gate_num",), sq_data.sel(mixer="err").values),
            },
            coords={"gate_num": sq_data.coords["x"].values},
        )
        label = sq_data.name

        # Run analysis (if needed for side effects)
        my_ana = Clifford1QRB(data, label)
        my_ana._start_analysis(plot=False)

        # Plot the raw STD data vs. gate number.
        ax.plot(data.coords["gate_num"].values, data["std"].values, "-", label=subfolder_name)

    plt.tight_layout()
    plt.show()

##################################
# Section 2: 2D Colormap Plots
##################################
# Here we loop over each folder, build a 2D array (rows = measurement index, columns = gate number),
# and then plot a colormap using imshow.

# Dictionary to hold the colormap data for each folder.
colormap_data = {}

for folder_name_str in folder_list:
    folder_path = Path(r"D:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\1QRB_2") / folder_name_str
    folder_names = [item.name for item in folder_path.iterdir() if item.is_dir()]
    std_list = []
    # Assume all files have the same gate number axis; we take it from the first file.
    gate_num_common = None

    print(f"Collecting colormap data for folder '{folder_name_str}' with {len(folder_names)} subfolders...")
    
    for subfolder_name in folder_names:
        file_path = folder_path / subfolder_name / "1QRB.nc"
        dataset = xr.open_dataset(str(file_path))
        sq_data = dataset[ro_name]
        gate_num = sq_data.coords["x"].values
        if gate_num_common is None:
            gate_num_common = gate_num
        data = xr.Dataset(
            {
                "p0": (("gate_num",), sq_data.sel(mixer="val").values),
                "std": (("gate_num",), sq_data.sel(mixer="err").values),
            },
            coords={"gate_num": gate_num},
        )
        # Append the STD data for this measurement (file)
        std_list.append(data["std"].values)
    
    # Build a 2D array: each row corresponds to one measurement file.
    std_2d = np.array(std_list)
    colormap_data[folder_name_str] = (gate_num_common, std_2d)

# Option 1: Plot each folder's colormap in its own figure
for folder_name_str, (gate_num, std_2d) in colormap_data.items():
    fig, ax = plt.subplots(1, figsize=(8, 6))
    # Set the extent so that the x-axis covers the range of gate numbers and the y-axis
    # covers the measurement index (0 to number of files)
    extent = [gate_num[0], gate_num[-1], 0, std_2d.shape[0]]
    im = ax.imshow(std_2d, aspect='auto', extent=extent, origin='lower', interpolation='nearest')
    ax.set_xlabel("Gate Number", fontsize=14)
    ax.set_ylabel("Measurement Index", fontsize=14)
    ax.set_title(f"2D Colormap: {folder_name_str}", fontsize=16)
    fig.colorbar(im, ax=ax, label="STD")
    plt.tight_layout()
    plt.show()

##################################
# Section 3: Difference 2D Colormap Plot
##################################
# Here we compute the difference between the two colormaps (independent - simultaneous)
# and plot the difference.

if "independent" in colormap_data and "simultaneous" in colormap_data:
    gate_num_indep, std_indep = colormap_data["independent"]
    gate_num_simul, std_simul = colormap_data["simultaneous"]
    
    # If necessary, you might need to check for matching dimensions. For now we assume they match.
    diff_std = std_indep - std_simul

    fig, ax = plt.subplots(1, figsize=(8, 6))
    extent = [gate_num_indep[0], gate_num_indep[-1], 0, diff_std.shape[0]]
    im = ax.imshow(diff_std, aspect='auto', extent=extent, origin='lower', interpolation='nearest', cmap='viridis')
    ax.set_xlabel("Gate Number", fontsize=14)
    ax.set_ylabel("Measurement Index", fontsize=14)
    ax.set_title("Difference Colormap (Independent - Simultaneous)", fontsize=16)
    fig.colorbar(im, ax=ax, label="STD Difference")
    plt.tight_layout()
    plt.show()
else:
    print("One or both folders ('independent', 'simultaneous') are missing from the colormap data.")

plt.tight_layout()
plt.show()
