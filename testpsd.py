import numpy as np
from scipy import signal

# 生成示例脑电信号
fs = 1000  # 采样率为1000Hz
t = np.arange(0, 1, 1/fs)  # 1秒的时间
eeg_signal = np.sin(2 * np.pi * 8 * t) + np.sin(2 * np.pi * 15 * t) + np.sin(2 * np.pi * 25 * t)

# 分解脑电信号为五个频段
frequencies, power_spectrum = signal.periodogram(eeg_signal, fs)
delta_power = np.trapz(power_spectrum[np.where((0.5 <= frequencies) & (frequencies <= 4))], frequencies[np.where((0.5 <= frequencies) & (frequencies <= 4))])
theta_power = np.trapz(power_spectrum[np.where((4 <= frequencies) & (frequencies <= 8))], frequencies[np.where((4 <= frequencies) & (frequencies <= 8))])
alpha_power = np.trapz(power_spectrum[np.where((8 <= frequencies) & (frequencies <= 13))], frequencies[np.where((8 <= frequencies) & (frequencies <= 13))])
beta_power = np.trapz(power_spectrum[np.where((13 <= frequencies) & (frequencies <= 30))], frequencies[np.where((13 <= frequencies) & (frequencies <= 30))])
gamma_power = np.trapz(power_spectrum[np.where((30 <= frequencies) & (frequencies <= 100))], frequencies[np.where((30 <= frequencies) & (frequencies <= 100))])

# 输出各个波段的功率
print("Delta Power:", delta_power)
print("Theta Power:", theta_power)
print("Alpha Power:", alpha_power)
print("Beta Power:", beta_power)
print("Gamma Power:", gamma_power)
