# plotting.py
import pandas as pd
import matplotlib.pyplot as plt
import os
def plot_fit(photons_sorted, delta_tot, fitted_delta, differentiated_data, fitted_differentiated, fr_value):
    # 創建兩個子圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # 第一個子圖：原始數據和擬合結果
    ax1.scatter(photons_sorted, delta_tot, label='Data', color='blue')
    ax1.plot(photons_sorted, fitted_delta, color='red', label='Fit')
    ax1.set_xscale('log')
    ax1.set_xlabel('N_{photons}')
    ax1.set_ylabel('Δ_tot')
    ax1.set_title(f'Fitting Result for fr={fr_value}')
    ax1.legend()

    # 第二個子圖：(d Q_i/d photon) * photon 數據及其擬合結果
    ax2.scatter(photons_sorted, differentiated_data, label='(d_delta/dphoton) * photon Data', color='green')
    ax2.plot(photons_sorted, fitted_differentiated, color='orange', label='Fit for (d_delta/dphoton) * photon')
    ax2.set_xscale('log')
    ax2.set_xlabel('N_{photons}')
    ax2.set_ylabel('(d_delta/dphoton) * photon')
    ax2.set_title(f'(d_delta/dphoton) * photon for fr={fr_value}')
    ax2.legend()

    # 調整子圖佈局
    plt.tight_layout()
    plt.show()

def plot_results(result_folder):
    # 讀取擬合結果 CSV 文件
    fitting_results_path = os.path.join(result_folder, 'fitting_results_with_derivative.csv')
    fitting_results = pd.read_csv(fitting_results_path)
    
    # 確保 'fr' 和擬合參數的欄位存在
    if 'fr' not in fitting_results.columns or 'delta_TLS' not in fitting_results.columns or 'N_sat' not in fitting_results.columns:
        raise ValueError("Fitting results CSV missing required columns.")
    
    # 計算 delta_TLS 的最大值和最小值
    delta_max = fitting_results['delta_max']
    delta_min = fitting_results['delta_min']
    delta_diff = delta_max - delta_min
    
    # 合併圖像
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 繪製 delta_TLS 和 delta_TLS max - min
    color = 'tab:blue'
    ax1.set_xlabel('Frequency (fr)')
    ax1.set_ylabel('delta_TLS', color=color)
    ax1.plot(fitting_results['fr'], fitting_results['delta_TLS'], 'o-', label='delta_TLS', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    
    color = 'tab:green'
    ax1.plot(fitting_results['fr'], delta_diff, 'd-', label='delta_TLS max - min', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 繪製 N_sat
    ax2 = ax1.twinx()  # 創建第二個 y 軸
    color = 'tab:red'
    ax2.set_ylabel('N_sat', color=color)
    ax2.plot(fitting_results['fr'], fitting_results['N_sat'], 's-', label='N_sat', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')

    # 設置圖例和標題
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()  # 使佈局更緊湊
    ax1.set_title('Frequency vs delta_TLS, delta_TLS max - min, and N_sat')
    
    # 儲存圖像
    output_path = os.path.join(result_folder, 'Combined_Plot.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.legend()
    plt.show()
    
    print(f"Plot has been saved to {output_path}")
