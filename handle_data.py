# 导入必要的库
import os
import numpy as np
import scipy.signal
import scipy.io as sio

# 定义文件路径参与者数量、通道数和采样频率。
file_path = 'D:/BaiduNetdiskDownload/MODMA/EEG_3channels_resting_lanzhou_2015/'
sub = 3
channels = 3
fs = 250
# 检查文件是否存在,如果存在则按顺序处理每个文件。
if os.path.exists(file_path):
    # 打开文件
    for i in range(sub):
        if i < 10:
            with open(file_path + "0201000" + str(i + 1) + "_still.txt", 'r') as file:
                # 读取文件内容
                eeg = file.read()
        else:
            with open(file_path + "020100" + str(i + 1) + "_still.txt", 'r') as file:
                # 读取文件内容
                eeg = file.read()
        # 这行代码将eeg文本按行分割，得到一个包含每行内容的列表。
        lines = eeg.splitlines()
        # 这行代码将列表中的每行内容按空格分割，得到一个包含每行元素的列表。
        data_list = [list(map(int, line.split())) for line in lines if line]
        # 这行代码将列表中的每行元素转换为numpy数组。
        eeg_data = np.array(data_list, dtype=int)   # (301740,3)
        eeg_data = eeg_data[-(90 * 250):, :]  # 这里修改成你的数据长度，这里是90(秒)的长度  22500,3
        eeg_data = eeg_data.reshape(-1, 250, 3)  # 数据格式（时间，采样率，电极数量）->1s时间窗
        eeg_data = eeg_data.transpose(0, 2, 1)  # 数据格式（时间，电极数量，采样率）
        freq_bands = 5

        # noinspection PyUnresolvedReferences# 计算功率谱密度
        (f, psd) = scipy.signal.welch(eeg_data, fs, nperseg=fs, window='hamming')
        # 计算各频段的平均功率值
        X = np.zeros((eeg_data.shape[0], channels, freq_bands))
        # delta band (4-7Hz)
        X[0:eeg_data.shape[0], 0:channels, 0] = np.mean(10 * np.log10(psd[0:eeg_data.shape[0], 0:channels, 4:8]),
                                                        axis=2)
        # slow alpha band (8-10Hz)
        X[0:eeg_data.shape[0], 0:channels, 0] = np.mean(10 * np.log10(psd[0:eeg_data.shape[0], 0:channels, 8:11]),
                                                        axis=2)
        # alpha band (8-12Hz)
        X[0:eeg_data.shape[0], 0:channels, 0] = np.mean(10 * np.log10(psd[0:eeg_data.shape[0], 0:channels, 8:13]),
                                                        axis=2)
        # beta band (13-30Hz)
        X[0:eeg_data.shape[0], 0:channels, 0] = np.mean(10 * np.log10(psd[0:eeg_data.shape[0], 0:channels, 13:31]),
                                                        axis=2)
        # gamma band (30-47Hz)
        X[0:eeg_data.shape[0], 0:channels, 0] = np.mean(10 * np.log10(psd[0:eeg_data.shape[0], 0:channels, 30:48]),
                                                        axis=2)

        # 定义label-自己定义正常人和抑郁症患者的标签
        label = np.ones((eeg_data.shape[0], 1))
    print("psd特征：", X.shape)
    print("psd特征：", X)
