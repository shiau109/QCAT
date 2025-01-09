import numpy as np
import matplotlib.pyplot as plt

# 定義一個時域函數（例如：正弦波加高斯波）
gate_time = 400
t = np.linspace(0, gate_time, gate_time, endpoint=False)  # 時間軸
# f_signal = np.sin(2 * np.pi * 1 * x) + 0.5 * np.cos(2 * np.pi * 3 * x)  # 組合波形
sfactor = 4
sigma = gate_time / sfactor
f_signal = np.sin(2 * np.pi * t)
# f_signal = np.exp(-((t - gate_time/2) ** 2) / (2 * sigma**2))
# 傅立葉變換
f_transform = np.fft.fft(f_signal)  # 傅立葉變換
frequencies = np.fft.fftfreq(len(t), d=(t[1] - t[0]))  # 計算頻率

positive_freq_indices = frequencies > 0  # 過濾條件
positive_frequencies = frequencies[positive_freq_indices]
positive_amplitude = np.abs(f_transform)[positive_freq_indices]  # 幅值
positive_phase = np.angle(f_transform)[positive_freq_indices]    # 相位

# 繪製原始信號
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(t, f_signal)
plt.title("Time Domain Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# 繪製頻域的幅值
plt.subplot(2, 2, 2)
plt.plot(positive_frequencies, positive_amplitude)
plt.title("Frequency Domain (Amplitude)")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")

# 繪製頻域的相位
plt.subplot(2, 2, 3)
plt.plot(positive_frequencies, positive_phase)
plt.title("Frequency Domain (Phase)")
plt.xlabel("Frequency")
plt.ylabel("Phase (radians)")

# 調整版面與顯示圖形
plt.tight_layout()
plt.show()