import matplotlib.pyplot as plt
import numpy as np

from qcat.analysis.qubit.clifford_1QRB import Clifford1QRB
import xarray as xr


from pathlib import Path


folder_list = ["independent","simultaneous"]
fig, ax = plt.subplots(1)
x = np.array([-0.08,-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08])
x = np.sqrt(8*0.2*27*abs(np.cos((x+0.115)/0.7*np.pi)))-0.2
all_y = []
ro_name = "q1_ro"
for folder_name_str in folder_list:
# Specify the folder path
    folder_path = Path(r"D:\Data\Qubit\5Q4C0430\20241121_DR3_5Q4C_0430#7_q2q3\TPS\1QRB_2\\"+folder_name_str)

# Get all folder names
    folder_names = [item.name for item in folder_path.iterdir() if item.is_dir()]

    print("Folders:", len(folder_names))
    r_g_list = []
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

        r_g_list.append( my_ana.fidelity["native_gate_fidelity"] )

    y = np.array(r_g_list)
    ax.plot( x, y*100,"o", label=folder_name_str)

    all_y.append(y)




# ax.set_title(f"{title} T2 Ramsey I data")

ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Coupler Frequency (GHz)", fontsize=20)
ax.set_ylabel("Native Gate Error Rate (%)", fontsize=20)
# ax.set_yscale("log")
# ax.set_ylim(0.002, 0.01)
ax.set_ylim(0.2, 1.0)

ax.set_xlim(5.1, 6.5)

# ax.plot( x, y2,"o", label="indipendent")
ax.legend()
plt.tight_layout()

plt.show()

fig, ax = plt.subplots(1)
ax.plot( x, (all_y[1]-all_y[0])*100,"o", label=folder_name_str)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel("Coupler Frequency (GHz)", fontsize=20)
ax.set_ylabel("Error Rate Difference (%)", fontsize=20)
ax.hlines(0, 5.1, 6.5, color="black", linestyle="--")

# ax.set_ylim(0.2, 1.0)
ax.set_xlim(5.1, 6.5)
plt.tight_layout()

plt.show()
