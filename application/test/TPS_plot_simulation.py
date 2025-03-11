import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from qcat.analysis.qubit.clifford_1QRB import Clifford1QRB

folder_list = ["independent","simultaneous"]
fig, ax = plt.subplots(1)
x = np.array([-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08])
coupler_freq = np.sqrt(8 * 0.16 *43.73 * abs(np.cos((x +0.015 +0.10) /0.612 * np.pi))) - 0.16

all_y = []
ro_name = "q1_ro"
for folder_name_str in folder_list:
# Specify the folder path
    folder_path = Path(r"D:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\1QRB_2\\"+folder_name_str)

# Get all folder names
    folder_names = [item.name for item in folder_path.iterdir() if item.is_dir()]

    print("Folders:", len(folder_names))
    r_g_list = []
    r_g_err_list = []

    for folder_name in folder_names:
        all_path = str(folder_path)+"\\"+folder_name+r"\1QRB.nc"
        dataset = xr.open_dataset(all_path)
        seperated_dataset = []

        

        sq_data = dataset[ro_name]
        sq_data.attrs = dataset.attrs
        sq_data.name = ro_name

        # change format (temp)
        data = xr.Dataset(
            {
                "p0": (("gate_num",), sq_data.sel(mixer="val").values),
                "std": (("gate_num",), sq_data.sel(mixer="err").values),
            },
            coords={"gate_num": sq_data.coords["x"].values },
        )
        label = sq_data.name
            
        my_ana = Clifford1QRB( data, label )
        my_ana._start_analysis( plot=False )

        r_g_list.append( my_ana.fidelity["native_gate_infidelity"] )
        r_g_err_list.append( my_ana.fidelity["native_gate_infidelity_err"] )
    y = np.array(r_g_list)
    ax.errorbar( coupler_freq, y*100, np.array(r_g_err_list),fmt='o', label=folder_name_str)
    print(r_g_err_list)
    all_y.append(y)



# Load the NetCDF dataset
ds = xr.open_dataset(r'd:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\simulation\RB_detuned_wozz_fast.nc')
ds1 = xr.open_dataset(r'd:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\simulation\RB_detuned_wzz_fast.nc')

# Check dataset variables (optional, for debugging)
print(ds)

# Plot 1: r_g vs. coupler_freq
plt.figure()
ax.plot(ds['coupler_freq'].values, ds['r_g'].values*100, marker='o', label='independent')
ax.plot(ds1['coupler_freq'].values, ds1['r_g'].values*100, marker='o', label='simultaneous')
ax.set_xlabel('coupler_freq')
ax.set_ylabel('r_g')
ax.set_title('r_g vs. coupler_freq')
ax.legend()


# Plot 2: chi vs. coupler_freq
plt.figure()
plt.plot(ds['coupler_freq'], ds['chi'], marker='o', label="chi")
plt.plot(ds['coupler_freq'], ds['crosstalk_freq'], marker='o', label="crosstalk_freq")
plt.plot(ds1['coupler_freq'], ds1['zz_interaction'], marker='o', label="zz_interaction")

plt.xlabel('coupler_freq')
plt.ylabel('chi')
plt.title('chi vs. coupler_freq')
plt.legend()

# Plot 3: corr_coupling vs. coupler_freq
plt.figure()
plt.plot(ds['coupler_freq'], ds['corr_coupling'], marker='o')
plt.xlabel('coupler_freq')
plt.ylabel('corr_coupling')
plt.title('corr_coupling vs. coupler_freq')
plt.grid(True)

# Plot 4: coupler_freq vs. flux
plt.figure()
plt.plot(ds['flux'], ds['coupler_freq'], marker='o')
plt.xlabel('flux')
plt.ylabel('coupler_freq')
plt.title('coupler_freq vs. flux')
plt.grid(True)

# Plot 4: coupler_freq vs. flux
plt.figure()
plt.plot(ds['flux'], ds['crosstalk_freq'], marker='o')
plt.xlabel('flux')
plt.ylabel('crosstalk_freq')
plt.title('coupler_freq vs. flux')
plt.grid(True)



plt.show()
