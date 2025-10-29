
import os
import json
from qcat.parser.qm_reader import load_xarray_h5
# Folder to check
folder_path = r"D:\github\ASQMDriver\data\6SFQ\2025-10-29\#220_02a_resonator_spectroscopy_121806"

# Build paths
file_path = os.path.join(folder_path, 'ds_raw.h5')
json_state_path = os.path.join(folder_path, 'quam_state\\state.json')


ds = load_xarray_h5(file_path)
with open(json_state_path, 'r') as f:
    json_dict = json.load(f)

print(ds)

qubit_name = ds.coords['qubit'].values
for q in qubit_name:

    freq = ds["full_freq"].sel(qubit=q).values
    print(freq.shape)
    s21_data = ds["I"].sel(qubit=q).values +1j*ds["Q"].sel(qubit=q).values
    print(s21_data.shape)
