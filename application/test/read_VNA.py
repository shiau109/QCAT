import pandas as pd
from qcat.utility.file_structure import *
from qcat.utility.io_translator import *
# from analysis.analysis_method import *
from qcat.resonator.photon_dep.res_data import *

# 1. Sample path setting
sample_name = "AS133_5Q4C_WOPR"
project_folder = r"d:\Data\resonator"   # empty string "" for relative path 

# 1.1 File structure setting
sample_root = f"{project_folder}/{sample_name}"

# Read all sheets
all_sheets_df = pd.read_excel(f"{project_folder}/{sample_name}/C1.xls", sheet_name=None)


# If reading all sheets, the result is a dictionary with sheet names as keys
for sheet_name, df in all_sheets_df.items():
    print(f"Sheet name: {sheet_name}")
    # print(df.head())  # Display the first few rows of the DataFrame
    # print(f"Processing {cav_label}")
    freq = df["F"].values
    s21 = df["I"].values +1j*df["Q"].values
    # s21 = np.exp(np.log10(np.abs(s21))*0.5)*np.exp(1j*np.angle(s21))
    resonator = ResonatorData(freq, s21)
    # Find cavity data (mat file) in the folder

    result, zdata_norm, fit_curve_norm  = resonator.fit()
    plot_data = np.array([[freq,zdata_norm,fit_curve_norm]])
    result_df = pd.DataFrame.from_dict([result])
    result_df.to_csv(f"{sample_root}/{sheet_name}.csv")
    print(result, plot_data.shape)
    plot_resonatorFitting([sheet_name], plot_data, sheet_name)
    plt.show()