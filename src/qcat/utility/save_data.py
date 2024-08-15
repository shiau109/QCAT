from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def save_npz( save_dir, file_name, output_data:dict, time_label:str|None="prefix", time_format:str="%Y%m%d_%H%M" ):

    save_path = f"{save_dir}/{_add_time_label(file_name, time_label, time_format)}.npz"
    np.savez(save_path, **output_data)

def save_nc( save_dir, file_name, output_data:xr.Dataset, time_label:str|None="prefix", time_format:str="%Y%m%d_%H%M" ):

    save_path = f"{save_dir}/{_add_time_label(file_name, time_label, time_format)}.nc"
    output_data.to_netcdf(save_path)


def save_fig( save_dir, file_name, time_label:str|None="prefix", time_format:str="%Y%m%d_%H%M" ):
    
    save_path = f"{save_dir}/{_add_time_label(file_name, time_label, time_format)}.png"
    figure = plt.gcf()
    figure.set_size_inches(16, 8)
    plt.tight_layout()
    plt.savefig(f"{save_path}", dpi = 500)

def _add_time_label( file_name, time_label:str|None="prefix", time_format:str="%Y%m%d_%H%M" ):
    save_time = str(datetime.now().strftime(time_format))

    match time_label:
        case "suffix":
            new_file_name = f"{file_name}_{save_time}"
        case "prefix":
            new_file_name = f"{save_time}_{file_name}"
        case _:
            new_file_name = f"{file_name}"  
    return new_file_name

def create_folder(save_dir, folder_name):
    # Combine fixed part with user input
    folder_name = f"Experiment_{folder_name}"
    full_path = os.path.join(save_dir, folder_name)

    if os.path.exists(full_path):
        print(f"Warning: The directory '{full_path}' already exists, creating separate folder")
        count = 0
        while os.path.exists(full_path):
            folder_name = f"Experiment_{folder_name}_{count}"
            full_path = os.path.join(save_dir, folder_name)
            count+=1
        os.makedirs(full_path, exist_ok=True)
        return full_path
    else:
        # Create the directory
        os.makedirs(full_path, exist_ok=True)
        print(f"Directory '{full_path}' created successfully.")
        return full_path
