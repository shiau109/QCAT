import os
import json
import xarray as xr
import numpy as np
import matplotlib as plt
from datetime import datetime


from qcat.utility.file_structure import *
from qcat.utility.io_translator import *
# from analysis.analysis_method import *
from qcat.visualization.photon_dep_loss import * 

root_fd = r"D:\Data\TWPA\V14W15D2"



from qcat.analysis.resonator.photon_dep.res_data import *
from qcat.utility.file_structure import check_file_extension, create_subfolder


# check_configure(f"{output_fd}", ["power_dep_fit"])
fn_bypass = r"d:\Data\TWPA\Bypass\Bypass_RO_30dBm_PF0MHz_PP0dBm_BB.nc"
fn_on = r"d:\Data\TWPA\V14W15D2\V14W15D2_RO_30dBm_PF7057MHz_PPm2188dBm_BB.nc"
fn_off = r"d:\Data\TWPA\V14W15D2\V14W15D2_RO_30dBm_PF0MHz_PP0dBm_BB.nc"
file_list = [fn_off,fn_on,fn_bypass]
legend_label = ["OFF","ON","Bypass"]

ref_label = "Bypass"
# file_list = ['liteVNA_80_-15.0.nc', 'liteVNA_80_-20.0.nc', 'liteVNA_80_-25.0.nc', 'liteVNA_80_-30.0.nc', 'liteVNA_80_-35.0.nc', 'liteVNA_80_-40.0.nc', 'liteVNA_80_-45.0.nc', 'liteVNA_80_-50.0.nc', 'liteVNA_80_-55.0.nc', 'liteVNA_80_-60.0.nc','liteVNA_110_-30.0.nc', 'liteVNA_110_-35.0.nc', 'liteVNA_110_-40.0.nc', 'liteVNA_110_-41.0.nc', 'liteVNA_110_-42.0.nc', 'liteVNA_110_-43.0.nc', 'liteVNA_110_-44.0.nc', 'liteVNA_110_-45.0.nc']

fig, ax = plt.subplots()
fig_dif, ax_dif = plt.subplots()

dataset = xr.open_dataset(f"{fn_bypass}")
freq = dataset.coords["frequency"].values/1e9
s21 = dataset["s21"].values[0]+1j*dataset["s21"].values[1]
power_ref = np.log10(np.abs(s21))*20

gain_list = []
for i in range(len(file_list)):
    f_name = file_list[i]
    dataset = xr.open_dataset(f"{f_name}")
    # print(dataset)
    freq = dataset.coords["frequency"].values/1e9
    s21 = dataset["s21"].values[0]+1j*dataset["s21"].values[1]
    power = np.log10(np.abs(s21))*20
    ax.plot(freq,power, label=legend_label[i])
    ax.legend()
    if legend_label[i] != ref_label:
        print(legend_label[i])
        gain = power-power_ref
        gain_list.append(gain)
        ax_dif.plot(freq,power-power_ref, label=legend_label[i])
        ax_dif.legend()

import pandas as pd
fig, ax = plt.subplots()
fig_dif, ax_dif = plt.subplots()

fn_bypass = r"d:\Data\TWPA\Bypass\Bypass.csv"
fn_on = r"d:\Data\TWPA\V14W15D2\V14W15D2_ON.csv"
fn_off = r"d:\Data\TWPA\V14W15D2\V14W15D2_OFF.csv"
file_list = [fn_off,fn_on,fn_bypass]
legend_label = ["OFF","ON","Bypass"]
ref_label = "Bypass"

df = pd.read_csv(fn_bypass, skiprows=44, names=["frequency","power"])
freq = df.iloc[:, 0].values/1e9

power_ref = df.iloc[:, 1].values

noise_list =[]
for i in range(len(file_list)):
    f_name = file_list[i]
# Load the CSV file into a DataFrame
    df = pd.read_csv(f_name, skiprows=44, names=["frequency","power"])
    freq = df.iloc[:, 0].values/1e9
    power = df.iloc[:, 1].values
    ax.plot(freq,power, label=legend_label[i])
    ax.legend()
     
    if legend_label[i] != ref_label:
        noise_diff = power-power_ref
        noise_list.append(noise_diff)
        ax_dif.plot(freq,noise_diff, label=legend_label[i])
        ax_dif.legend()

fig, ax = plt.subplots()
ax.plot(freq,gain_list[1]-noise_list[1], label=legend_label[i])


plt.show()



