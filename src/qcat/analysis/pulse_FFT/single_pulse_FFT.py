import numpy as np
import matplotlib.pyplot as plt

# 定義一個時域函數（例如：正弦波加高斯波）
gate_time = 400 # ns
IF_freq = 50 # MHz
pulse_num = 10

T = gate_time/1000
t = np.linspace(0, T, gate_time, endpoint=False)  # 時間軸

sfactor = 4
sigma = T / sfactor
# Square
# envelope = np.ones(gate_time)
# Gaussian
Gaussina0 = np.exp(-((t[0] - T/2) ** 2) / (2 * sigma**2))
# envelope = np.exp(-((t - T/2) ** 2) / (2 * sigma**2)) - Gaussina0
# Parabola cos square
# envelope = (0.318 - 0.005 * (t-T/2)**2)*(np.cos((np.pi/T)*(t-T/2)))**2
# Cos
envelope = 0.5 * (1 - np.cos(2 * np.pi * t / T))
# Sinc

IF_pulse = np.cos(2 * np.pi * IF_freq * t)
full_envelope = []
full_IF_pulse = []
full_t = []
for i in range(pulse_num):
    full_envelope = np.append(full_envelope, envelope)
    full_IF_pulse = np.append(full_IF_pulse, IF_pulse)
    full_t = np.append(full_t, t+T*i)
# IF_pulse = 1
f_signal = full_IF_pulse * full_envelope # 組合波形


# 傅立葉變換
f_transform = np.fft.fft(f_signal)  # 傅立葉變換
frequencies = np.fft.fftfreq(len(t)*pulse_num, d=(t[1] - t[0]))  # 計算頻率

positive_freq_indices = frequencies > 0  # 過濾條件
positive_frequencies = frequencies[positive_freq_indices]
positive_amplitude = np.abs(f_transform)[positive_freq_indices]  # 幅值
positive_phase = np.angle(f_transform)[positive_freq_indices]    # 相位

# 繪製原始信號
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(full_t, f_signal)
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