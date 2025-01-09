# main.py
import os
import pandas as pd
from qcat.analysis.resonator.photon_dep.TLS_loss_analysis import *
from qcat.visualization.TLS_loss_plot import *


# 指定資料夾路徑
result_folder = r"C:\Users\ASUS\Documents\python training\新增資料夾\ITRI367_resmxv2\result"  # 替換為你的result資料夾路徑

# 執行分析
fitting_results = analyze_data(result_folder)

# 遍歷結果，調用畫圖功能
for subdir, dirs, files in os.walk(result_folder):
    for file in files:
        if file == 'refined_result.csv':  # 檢查文件名稱
            file_path = os.path.join(subdir, file)
            
            # 讀取 CSV 文件
            df = pd.read_csv(file_path)
            
            # 提取數據
            photons = df['photons'].values
            Qi_dia_corr = df['Qi_dia_corr'].values

            # 根據 photons 排序
            sorted_indices = np.argsort(photons)
            photons_sorted = photons[sorted_indices]
            Qi_dia_corr_sorted = Qi_dia_corr[sorted_indices]
            delta_tot = 1 / Qi_dia_corr_sorted

            # 提取擬合參數
            fr_value = df['fr'].iloc[0]
            fit_row = fitting_results[fitting_results['fr'] == fr_value]

            if not fit_row.empty:
                delta_TLS, N_sat, alpha, delta_0 = fit_row.iloc[0][['delta_TLS', 'N_sat', 'alpha', 'delta_0']]
                # 計算擬合曲線
                fitted_delta = delta_TLS / np.sqrt(1 + (photons_sorted / N_sat)**alpha) + delta_0

                # 畫圖
                plot_fit(photons_sorted, delta_tot, fitted_delta, fr_value)
