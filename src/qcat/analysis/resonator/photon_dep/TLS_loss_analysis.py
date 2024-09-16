# analysis.py
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# 定義擬合的公式
def model_func(N_photons, delta_TLS, N_sat, alpha, delta_0):
    return delta_TLS / np.sqrt(1 + (N_photons / N_sat)**alpha) + delta_0

# 分析函數：處理擬合和結果保存
def analyze_data(result_folder):
    # 建立一個空的DataFrame來存儲所有擬合結果
    fitting_results = pd.DataFrame(columns=['fr', 'delta_TLS', 'N_sat', 'alpha', 'delta_0'])

    # 遍歷result資料夾內的所有子資料夾
    for subdir, dirs, files in os.walk(result_folder):
        for file in files:
            if file == 'refined_result.csv':  # 檢查文件名稱
                file_path = os.path.join(subdir, file)
                
                # 讀取 CSV 文件
                df = pd.read_csv(file_path)
                
                # 提取數據
                Qi_dia_corr = df['Qi_dia_corr'].values
                photons = df['photons'].values
                Qi_dia_corr_err = df['Qi_dia_corr_err'].values  # 提取誤差數據

                # 根據 photons 排序
                sorted_indices = np.argsort(photons)
                photons_sorted = photons[sorted_indices]
                Qi_dia_corr_sorted = Qi_dia_corr[sorted_indices]
                Qi_dia_corr_err_sorted = Qi_dia_corr_err[sorted_indices]
                
                # 計算 delta_tot
                delta_tot = 1 / Qi_dia_corr_sorted

                # 設置初始猜測值
                delta_0_initial = np.min(delta_tot)
                delta_TLS_initial = np.max(delta_tot) - delta_0_initial
                alpha_initial = 1.0  # 在 0 到 2 之間選擇一個值
                N_sat_initial = 1e0  # 假設的初始值

                # 提取 fr 欄位的第一個值
                fr_value = df['fr'].iloc[0]

                # 擬合數據，加權擬合使用誤差的倒數平方
                try:
                    popt, pcov = curve_fit(
                        model_func,
                        photons_sorted,
                        delta_tot,
                        p0=[delta_TLS_initial, N_sat_initial, alpha_initial, delta_0_initial],
                        sigma=Qi_dia_corr_err_sorted,
                        absolute_sigma=True
                    )
                    delta_TLS, N_sat, alpha, delta_0 = popt

                    # 保存擬合結果
                    new_row = pd.DataFrame({
                        'fr': [fr_value],
                        'delta_TLS': [delta_TLS],
                        'N_sat': [N_sat],
                        'alpha': [alpha],
                        'delta_0': [delta_0]
                    })
                    fitting_results = pd.concat([fitting_results, new_row], ignore_index=True)

                except RuntimeError as e:
                    print(f"Fitting failed for {file_path}: {e}")

    # 將擬合結果保存到一個新的 CSV 文件
    output_file = os.path.join(result_folder, 'fitting_results.csv')
    fitting_results.to_csv(output_file, index=False)
    print(f"所有擬合結果已保存至 {output_file}")

    return fitting_results
