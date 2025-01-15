import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# 定義擬合的公式
def diff_model_func(N_photons, delta_TLS, N_sat, alpha):
    term1 = (N_photons / N_sat) ** alpha
    return -alpha * delta_TLS * (term1 / (2 * (term1 + 1) ** (3/2)))

# 計算 d(Q_i)/d(photon)
def calculate_derivative(photons, delta_tot):
    derivative = np.gradient(delta_tot, photons)  # 使用數值微分計算 d(delta_tot)/d(photon)
    return derivative

# 剃除沒有 error 或 error 過大的點，同時刪除 Qi 為負的點
def filter_data(photons, delta_tot, errors, max_error_threshold=1e6):
    valid_indices = np.where((errors > 0) & (errors < max_error_threshold) & (delta_tot > 0))
    return photons[valid_indices], delta_tot[valid_indices], errors[valid_indices]

# 根據 delta_tot 的最大值和最小值計算 N_sat 初始值
def find_closest_photon_number(photons, delta_tot):
    delta_max = np.max(delta_tot)
    delta_min = np.min(delta_tot)
    delta_N_c = (delta_max - delta_min) / 2**(1/2) + delta_min

    # 找到最接近 delta_N_c 的 delta_tot 的索引
    closest_index = np.argmin(np.abs(delta_tot - delta_N_c))
    
    # 對應的 photon number 作為 N_sat 初始值
    return photons[closest_index]

# 計算殘差平方和 (RSS)
def calculate_rss(fitted_values, actual_values, errors):
    residuals = (actual_values - fitted_values) / errors
    rss = np.sum(residuals**2)
    return rss

# 分析函數：處理擬合和結果保存
def analyze_data(result_folder):
    # 建立一個空的 DataFrame 來存儲所有擬合結果
    fitting_results = pd.DataFrame(columns=['fr', 'delta_TLS', 'N_sat', 'alpha', 'delta_0', 'delta_max', 'delta_min'])

    # 遍歷 result 資料夾內的所有子資料夾
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

                # 剃除 Qi 為負的點，以及誤差過大的點或沒有誤差的點
                max_error_threshold = 1e9  # 可以根據需要設置誤差閾值
                photons_filtered, delta_tot_filtered, errors_filtered = filter_data(
                    photons_sorted, delta_tot, Qi_dia_corr_err_sorted, max_error_threshold
                )

                # 計算微分後的數據
                delta_tot_derivative = calculate_derivative(photons_filtered, delta_tot_filtered)

                # 計算 (d(Q_i)/d(photon)) * photon
                derivative_times_photon = delta_tot_derivative * photons_filtered

                # 設置初始猜測值
                delta_TLS_initial = np.max(delta_tot_filtered) - np.min(delta_tot_filtered)
                alpha_initial = 1.0  # 初始 alpha 值
                N_sat_initial = find_closest_photon_number(photons_filtered, delta_tot_filtered)

                # 擬合公式 (使用 d(Q_i)/d(photon) * photon 的數據進行擬合)
                try:
                    popt, pcov = curve_fit(
                        diff_model_func,
                        photons_filtered,
                        derivative_times_photon,
                        p0=[delta_TLS_initial, N_sat_initial, alpha_initial],
                        sigma=errors_filtered,
                        absolute_sigma=True,
                        maxfev=50000000
                    )

                    # 擬合成功，提取參數
                    delta_TLS, N_sat, alpha = popt
                    fitted_values = diff_model_func(photons_filtered, delta_TLS, N_sat, alpha)
                    rss = calculate_rss(fitted_values, derivative_times_photon, errors_filtered)

                    delta_max = np.max(delta_tot_filtered)
                    delta_min = np.min(delta_tot_filtered)

                    # 保存擬合結果
                    fr_value = df['fr'].iloc[0]
                    new_row = pd.DataFrame({
                        'fr': [fr_value],
                        'delta_TLS': [delta_TLS],
                        'N_sat': [N_sat],
                        'alpha': [alpha],
                        'delta_max': [delta_max],
                        'delta_min': [delta_min]
                    })
                    fitting_results = pd.concat([fitting_results, new_row], ignore_index=True)

                except RuntimeError as e:
                    print(f"Fitting failed for {file_path}: {e}")

    # 將擬合結果保存到一個新的 CSV 文件
    output_file = os.path.join(result_folder, 'fitting_results_with_derivative.csv')
    fitting_results.to_csv(output_file, index=False)
    print(f"所有擬合結果已保存至 {output_file}")

    return fitting_results
