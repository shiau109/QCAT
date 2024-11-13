import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
N_sat_lowerbound = 1e-2
N_sat_upperbound = 1e4
alpha_lowerbound = 0.2
# 定義擬合的公式
def model_func(N_photons_log, delta_TLS, N_sat, alpha, delta_0):
    N_photons = 10**N_photons_log
    return delta_TLS / np.sqrt(1 + (N_photons / N_sat)**alpha) + delta_0

def model_func1(N_photons_log, delta_TLS, alpha, delta_0):
    N_photons = 10**N_photons_log
    return delta_TLS / np.sqrt(1 + (N_photons / N_sat_lowerbound)**alpha) + delta_0

def model_func2(N_photons_log, delta_TLS, alpha, delta_0):
    N_photons = 10**N_photons_log
    return delta_TLS / np.sqrt(1 + (N_photons / N_sat_upperbound)**alpha) + delta_0

def model_func3(N_photons_log, delta_TLS, N_sat, delta_0):
    N_photons = 10**N_photons_log
    return delta_TLS / np.sqrt(1 + (N_photons / N_sat)**alpha_lowerbound) + delta_0
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
                max_error_threshold = 1e8  # 可以根據需要設置誤差閾值
                photons_filtered, delta_tot_filtered, errors_filtered = filter_data(
                    photons_sorted, delta_tot, Qi_dia_corr_err_sorted, max_error_threshold
                )
                log_photons_filtered = np.log10(photons_filtered)

                # 設置初始猜測值，使用 delta_tot 的前 1/4 小和前 1/4 大的值
                n_quarter = len(delta_tot_filtered) // 15
                delta_0_initial = delta_tot_filtered[-n_quarter]  # 前1/4小的值作為delta_0
                delta_TLS_initial = delta_tot_filtered[n_quarter] - delta_0_initial  # 前1/4大的值減去前1/4小的值作為delta_TLS
                alpha_initial = 1.0  # 初始 alpha 值

                # 根據 delta_tot 的最大值和最小值，找到對應的 photon number 作為 N_sat 初始值
                N_sat_initial = find_closest_photon_number(photons_filtered, delta_tot_filtered)

                # 設置三組 N_sat 的初始值
                N_sat_initials = [0.1 * N_sat_initial, N_sat_initial, 10 * N_sat_initial]

                # 提取 fr 欄位的第一個值
                fr_value = df['fr'].iloc[0]

                best_fit_params = None
                best_rss = np.inf

                for N_sat_initial in N_sat_initials:
                    try:
                        # 擬合數據，加權擬合使用誤差的倒數平方
                        popt, pcov = curve_fit(
                            model_func,
                            log_photons_filtered,
                            delta_tot_filtered,
                            p0=[delta_TLS_initial, N_sat_initial, alpha_initial, delta_0_initial],
                            sigma=errors_filtered,
                            absolute_sigma=True,
                            maxfev=50000000  # 增加最大允許的迭代次數
                        )

                        # 擬合成功，提取參數
                        fitted_values = model_func(log_photons_filtered, *popt)
                        rss = calculate_rss(fitted_values, delta_tot_filtered, errors_filtered)

                        if rss < best_rss:
                            best_rss = rss
                            best_fit_params = popt

                    except RuntimeError as e:
                        print(f"Fitting failed for {file_path} with N_sat_initial = {N_sat_initial}: {e}")

                # 檢查 N_sat，如果小於 10^-3，將 N_sat 設為 10^-3 並重新擬合
                if best_fit_params is not None:
                    delta_TLS, N_sat, alpha, delta_0 = best_fit_params

                    if N_sat < N_sat_lowerbound:
                        print(f"N_sat < {N_sat_lowerbound} for {file_path}, refitting with N_sat = {N_sat_lowerbound}")
                        N_sat = N_sat_lowerbound  # 將 N_sat 設為 10^-3

                        try:
                            popt, pcov = curve_fit(
                                model_func1,
                                log_photons_filtered,
                                delta_tot_filtered,
                                p0=[delta_TLS, alpha, delta_0],  # 使用設為 10^-3 的 N_sat 重新擬合
                                sigma=errors_filtered,
                                absolute_sigma=True,
                                maxfev=50000000
                            )
                            popt = np.insert(popt, 1, N_sat)
                            best_fit_params = popt
                        except RuntimeError as e:
                            print(f"Refitting failed for {file_path} with N_sat = {N_sat_lowerbound}: {e}")
                    
                    if N_sat > N_sat_upperbound:
                        print(f"N_sat > {N_sat_upperbound} for {file_path}, refitting with N_sat = {N_sat_upperbound}")
                        N_sat = N_sat_upperbound  # 將 N_sat 設為 10^3

                        try:
                            popt, pcov = curve_fit(
                                model_func2,
                                log_photons_filtered,
                                delta_tot_filtered,
                                p0=[delta_TLS, alpha, delta_0],  # 使用設為 10^3 的 N_sat 重新擬合
                                sigma=errors_filtered,
                                absolute_sigma=True,
                                maxfev=50000000
                            )
                            popt = np.insert(popt, 1, N_sat)
                            best_fit_params = popt
                        except RuntimeError as e:
                            print(f"Refitting failed for {file_path} with N_sat = {N_sat_upperbound}: {e}")
                    
                    if alpha < alpha_lowerbound:
                        print(f"alpha < {alpha_lowerbound} for {file_path}, refitting with alpha = {alpha_lowerbound}")
                        alpha = alpha_lowerbound  # 將 N_sat 設為 10^3

                        try:
                            popt, pcov = curve_fit(
                                model_func3,
                                log_photons_filtered,
                                delta_tot_filtered,
                                p0=[delta_TLS, N_sat, delta_0],  # 使用設為 10^3 的 N_sat 重新擬合
                                sigma=errors_filtered,
                                absolute_sigma=True,
                                maxfev=50000000
                            )
                            popt = np.insert(popt, 2, alpha)
                            best_fit_params = popt
                        except RuntimeError as e:
                            print(f"Refitting failed for {file_path} with alpha = {alpha_lowerbound}: {e}")
                    
                    delta_TLS, N_sat, alpha, delta_0 = best_fit_params
                    delta_max = np.max(delta_tot_filtered)
                    delta_min = np.min(delta_tot_filtered)

                    # 保存擬合結果
                    new_row = pd.DataFrame({
                        'fr': [fr_value],
                        'delta_TLS': [delta_TLS],
                        'N_sat': [N_sat],
                        'alpha': [alpha],
                        'delta_0': [delta_0],
                        'delta_max': [delta_max],
                        'delta_min': [delta_min]
                    })
                    fitting_results = pd.concat([fitting_results, new_row], ignore_index=True)

    # 將擬合結果保存到一個新的 CSV 文件
    output_file = os.path.join(result_folder, 'fitting_results.csv')
    fitting_results.to_csv(output_file, index=False)
    print(f"所有擬合結果已保存至 {output_file}")

    return fitting_results
