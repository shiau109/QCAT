
import os
from qcat.parser.qm_reader import load_xarray_h5, repetition_data
import json
from lmfit import Model
import numpy as np

def plot_state_vs_basis(sq_sata, title_suffix=None):
    import matplotlib.pyplot as plt
    basis = sq_sata.coords['basis'].values
    state_on = sq_sata['state'].sel(ctrl_switch=True).values
    state_off = sq_sata['state'].sel(ctrl_switch=False).values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(basis, state_on, 'o-', label='ctrl_switch=on')
    ax.plot(basis, state_off, 'o-', label='ctrl_switch=off')
    ax.set_xlabel('basis')
    ax.set_ylabel('state')
    ax.set_title(f'state vs basis{title_suffix or ""}')
    ax.legend()
    fig.tight_layout()
    return fig
def plot_conditional_phase(op_times, amp_ratio, phase_diff, sine_shift_on, sine_shift_off, title_suffix=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].scatter(op_times, amp_ratio)
    axes[0].set_xlabel('operation_times')
    axes[0].set_ylabel('amp_ratio')
    axes[0].set_title(f'Amplitude Ratio vs Operation Times{title_suffix or ""}')

    axes[1].scatter(op_times, phase_diff)
    axes[1].set_xlabel('operation_times')
    axes[1].set_ylabel('phase_diff')
    axes[1].set_title(f'Phase Difference vs Operation Times{title_suffix or ""}')
    fig.tight_layout()

    # Create second figure for sine_shift
    import numpy as np
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
    axes2[0].scatter(op_times, sine_shift_on, label='sine_shift_on')
    axes2[0].scatter(op_times, sine_shift_off, label='sine_shift_off')
    axes2[0].set_xlabel('operation_times')
    axes2[0].set_ylabel('sine_shift')
    axes2[0].set_title(f'Sine Shift (on/off) vs Operation Times{title_suffix or ""}')
    axes2[0].legend()

    # Sort by op_times before calculating differences
    sort_idx = np.argsort(op_times)
    op_times_sorted = np.array(op_times)[sort_idx]
    sine_shift_on_sorted = np.array(sine_shift_on)[sort_idx]
    sine_shift_off_sorted = np.array(sine_shift_off)[sort_idx]
    diff_on = np.diff(sine_shift_on_sorted)
    diff_off = np.diff(sine_shift_off_sorted)
    # Wrap to [-pi, pi]
    diff_on = np.where(diff_on > np.pi, diff_on - 2 * np.pi, diff_on)
    diff_on = np.where(diff_on < -np.pi, diff_on + 2 * np.pi, diff_on)
    diff_off = np.where(diff_off > np.pi, diff_off - 2 * np.pi, diff_off)
    diff_off = np.where(diff_off < -np.pi, diff_off + 2 * np.pi, diff_off)
    axes2[1].plot(op_times_sorted[1:], diff_on-diff_on[0], 'o', label='diff_on')
    axes2[1].plot(op_times_sorted[1:], diff_off-diff_off[0], 'o', label='diff_off')
    axes2[1].set_xlabel('operation_times')
    axes2[1].set_ylabel('phase diff')
    axes2[1].set_title(f'Phase Difference Between Steps{title_suffix or ""}')
    axes2[1].legend()
    fig2.tight_layout()

    return fig, fig2

def fit_cosine_const(x, y):

    from lmfit.models import ConstantModel, SineModel
    sine_model = SineModel(prefix='sine_')
    const_model = ConstantModel(prefix='const_')
    model = sine_model + const_model
    params = model.make_params()
    params['sine_amplitude'].set(value=0.5, min=0, max=np.max(y)-np.min(y))
    params['sine_frequency'].set(value=2*np.pi, vary=False)
    params['sine_shift'].set(value=0, min=-np.pi, max=np.pi)
    params['const_c'].set(value=np.mean(y), min=2*np.min(y), max=2*np.max(y))
    result = model.fit(y, params, x=x)
    results = {
        'best_values': result.best_values,
        'result': result
    }
    return results

def get_conditional_phase(sq_sata):
    x = sq_sata.coords['basis'].values
    results = {}
    for val in [True, False]:
        label = "on" if val else "off"
        print(val, label)
        y = sq_sata['state'].sel(ctrl_switch=val).values
        result = fit_cosine_const(x, y)
        results[label] = result
    phase_diff = results["on"]['best_values']['sine_shift'] - results["off"]['best_values']['sine_shift']
    # Wrap phase_diff to [-pi, pi]
    if phase_diff < -np.pi:
        phase_diff += 2 * np.pi
    elif phase_diff > np.pi:
        phase_diff -= 2 * np.pi
    results["phase_diff"] = phase_diff
    results["amp_ratio"] = (results["on"]['best_values']['sine_amplitude'] / results["off"]['best_values']['sine_amplitude'])
    return results

base_dir = r'd:\data\CZ_repeat'
dataset_list = []

for root, dirs, files in os.walk(base_dir):
    if 'ds_raw.h5' in files and 'node.json' in files:
        file_path = os.path.join(root, 'ds_raw.h5')
        json_path = os.path.join(root, 'node.json')
        try:
            ds = load_xarray_h5(file_path)
            with open(json_path, 'r') as f:
                json_dict = json.load(f)
            dataset_list.append((ds, json_dict))
            print(f"Loaded: {file_path}, loaded node.json")
        except Exception as e:
            print(f"Failed to load {file_path} or {json_path}: {e}")

print(f"Total datasets loaded: {len(dataset_list)}")


import xarray as xr
import matplotlib.pyplot as plt

# Collect datasets and operation_times
ds_list = []
op_times_list = []
for ds, json_dict in dataset_list:
    ds_list.append(ds)
    op_times_list.append(json_dict["data"]["parameters"]["model"]['operation_times'])

# Stack datasets along new dimension 'operation_times'
combined_ds = xr.concat(ds_list, dim='operation_times')
combined_ds = combined_ds.assign_coords(operation_times=('operation_times', op_times_list))


# Now use repetition_data on combined_ds
for sq_sata in repetition_data(combined_ds, repetition_dim="qubit"):
    op_times = sq_sata.coords['operation_times'].values
    print(sq_sata)
    sq_sata_list = repetition_data(sq_sata, repetition_dim="operation_times")

    amp_ratio = []
    phase_diff = []
    sine_shift_on = []
    sine_shift_off = []
    for i, single_sq_sata in enumerate(sq_sata_list):
        if i == 0 or i==1 or i==8:
            plot_state_vs_basis(single_sq_sata, title_suffix=f' (qubit={str(single_sq_sata.coords["qubit"].values.item())})')
        fit_results = get_conditional_phase(single_sq_sata)
        amp_ratio.append(fit_results['amp_ratio'])
        phase_diff.append(fit_results['phase_diff'])
        sine_shift_on.append(fit_results['on']['best_values']['sine_shift'])
        sine_shift_off.append(fit_results['off']['best_values']['sine_shift'])
    print(sine_shift_on)
    fig = plot_conditional_phase(op_times, amp_ratio, phase_diff, sine_shift_on, sine_shift_off)
plt.show()