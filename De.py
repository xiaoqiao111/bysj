"""这是一个用于计算差分熵（DE）和功率谱密度（PSD）的函数DE_PSD。该函数接受包含原始数据和频谱参数的输入，并返回计算好的功率谱密度和差分熵。"""
import os
import numpy as np
import math
import pickle
from scipy.signal import butter, lfilter
from scipy.integrate import simps
import scipy
import scipy.io as sio
from scipy.fftpack import fft, ifft

def DE_PSD(data, stft_para):
    '''
    compute DE and PSD
    --------
    input:  data [n*m]          n 电极，m 时间点               40*8064
            stft_para.stftn     频域采样率       4
            stft_para.fStart    每个频率带的起始频率
            stft_para.fEnd      每个频率带的结束频率
            stft_para.window    每个采样点的窗口长度（秒） 256
            stft_para.fs        原始频率                  128
    output: psd,DE [n*l*k]        n 电极，l 窗口，k 频率带
    '''
    # initialize the parameters
    STFTN = stft_para['stftn']     #4
    fStart = stft_para['fStart']   #[4, 8, 14, 22, 30]
    fEnd = stft_para['fEnd']      #[7, 13, 21, 29, 45]
    fs = stft_para['fs']           #128
    window = stft_para['window']    #256
    # 计算窗口长度
    WindowPoints = fs * window    # 256*128=32768

    fStartNum = np.zeros([len(fStart)], dtype=int)       #[0 0 0 0 0]
    fEndNum = np.zeros([len(fEnd)], dtype=int)             #[0 0 0 0 0]
    for i in range(0, len(stft_para['fStart'])):         # i = 0,1,2,3,4
        fStartNum[i] = int(fStart[i] / fs * STFTN)
        fEndNum[i] = int(fEnd[i] / fs * STFTN)

    # print(fStartNum[0],fEndNum[0])
    n = data.shape[0]         # 40
    m = data.shape[1]         # 32

    # print(m,n,l)
    psd = np.zeros([n, len(fStart)])   #psd(40,5)
    de = np.zeros([n, len(fStart)])    #de(40,5)
    # Hanning window
    Hlength = window * fs   # 63*128=8064
    # Hwindow=hanning(Hlength)
    Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength + 1)) for n in range(1, Hlength + 1)])

    WindowPoints = fs * window
    # 将原始数据截取为n个数据点
    dataNow = data[0:n]
    for j in range(0, n):
        temp = dataNow[j]       #32,8064
        print(temp.shape)
        print(Hwindow.shape)
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata, STFTN)
        magFFTdata = abs(FFTdata[0:int(STFTN / 2)])
        for p in range(0, len(fStart)):
            E = 0
            E_log = 0
            for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                E = E + magFFTdata[p0] * magFFTdata[p0]
                # E_log = E_log + log2(magFFTdata(p0)*magFFTdata(p0)+1)
            E = E / (fEndNum[p] - fStartNum[p] + 1)
            psd[j][p] = E
            de[j][p] = math.log(100 * E, 2)
            # de(j,i,p)=log2((1+E)^4)
    return psd, de
stft_para = {
    'stftn': 4,
    'fStart': [4, 8, 14, 22, 30],  # 以列表形式提供每个频率段的起始频率
    'fEnd': [7, 13, 21, 29, 45],    # 以列表形式提供每个频率段的结束频率
    'window': 250,
    'fs': 250
}

def eeg_power_band(epochs):

    # 特定频带
    FREQ_BANDS = {"delta": [0.5, 4],
                  "theta": [4, 8],
                  "alpha": [8,21],
                  "sigma": [21, 30],
                  "beta": [30, 45]}
    spectrum = epochs.compute_psd(method='welch', picks='eeg', fmin=0.5, fmax=45, n_fft=64, n_overlap=10)
    psds, freqs = spectrum.get_data(return_freqs=True)
    # 归一化 PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

def extract_power_spectral_feature(eeg_data, state):
    # 从 EEG 信号中提取功率谱特征
    #       eeg_data:   [channels, frames] 的 EEG 信号数据
    #       srate:      int, 采样率
    #       power_features:   [1, n_features] 的特征向量
    # 计算各个节律频带的信号功率
    frequencies, psd = scipy.signal.welch(eeg_data, fs= state, window='hann', nperseg=255, noverlap=128)

    pt = []
    pa = []
    pl = []
    ph = []
    pg = []
    for i in range(3):
        power_theta = np.sum(psd[i,:][(frequencies >=4) & (frequencies <= 8)])
        pt.append(power_theta)
        power_alpha = np.sum(psd[i,:][(frequencies >= 8) & (frequencies <= 13)])
        pa.append(power_alpha)
        power_l_beta = np.sum(psd[i,:][(frequencies >= 13) & (frequencies <= 21)])
        pl.append(power_l_beta)
        power_h_beta = np.sum(psd[i,:][(frequencies >= 21) & (frequencies <= 30)])
        ph.append(power_h_beta)
        power_gamma = np.sum(psd[i,:][(frequencies >= 30) & (frequencies <= 45)])
        pg.append(power_gamma)
    """theta = [4, 8]
    alpha = [8, 13]
    l_beta = [13, 21]
    h_beta = [21, 30]
    gamma = [30, 45]
    # 计算频率分辨率
    freq_res = frequencies[1] - frequencies[0]"""
    # psd 为一系列离散值，无法直接积分，因此采用 simps 做抛物线近似
    """power_theta = np.logical_and(frequencies >= 4 ,frequencies <= 8)
    power_alpha = np.logical_and(frequencies >= 8, frequencies <= 13)
    power_l_beta = np.logical_and(frequencies >= 13, frequencies <= 21)
    power_h_beta = np.logical_and(frequencies >= 21 , frequencies <= 30)
    power_gamma = np.logical_and(frequencies >= 30, frequencies <= 45)
    band_power_theta = sum(psd[power_theta])
    band_power_alpha = sum(psd[power_alpha])
    band_power_l_beta = sum(psd[power_l_beta])
    band_power_h_beta = sum(psd[power_h_beta])
    band_power_gamma = sum(psd[power_gamma])"""
    """
    theta_power = np.trapz(psd[np.where((4 <= frequencies) & (frequencies <= 8))],frequencies[np.where((4 <= frequencies) & (frequencies <= 8))])
    alpha_power = np.trapz(psd[np.where((8 <= frequencies) & (frequencies <= 13))],frequencies[np.where((8 <= frequencies) & (frequencies <= 13))])
    l_beta_power = np.trapz(psd[np.where((13 <= frequencies) & (frequencies <= 21))],frequencies[np.where((13 <= frequencies) & (frequencies <= 21))])
    h_beta_power = np.trapz(psd[np.where((21 <= frequencies) & (frequencies <= 30))],frequencies[np.where((21 <= frequencies) & (frequencies <= 30))])
    gamma_power = np.trapz(psd[np.where((30 <= frequencies) & (frequencies <= 45))],frequencies[np.where((30 <= frequencies) & (frequencies <= 45))])
    """
    feature = np.vstack([pt, pa, pl, ph, pg])
    return feature



foldername = "D:\BaiduNetdiskDownload\MODMA\EEG_3channels_resting_lanzhou_2015"
file_list = os.listdir(foldername)  # 返回指定目录中的文件和子目录的名称列表。
features = []
channels = 3
for file_name in file_list:
    file_path = os.path.join(foldername, file_name)  # 拼接目录和文件名
    with open(file_path, 'r') as file:
        eeg = file.read()
        lines = eeg.splitlines()
        data_list = [list(map(int, line.split())) for line in lines if line]
        eeg_data = np.array(data_list, dtype=np.int64)
        eeg_data = eeg_data[-(250*60):, :]
        # eeg_data = eeg_data.reshape(-1, 250, 3)
        eeg_data1 = np.transpose(eeg_data)
        # eeg_data1 = eeg_data1.reshape(3, -1)
        # 输入的信号数据，格式为（时间，通道，采样点）的三维数组。
        feature1 = extract_power_spectral_feature(eeg_data1,state=250)
        features.append(feature1)
        #print(feature1.shape)  # (5, 1)

        # pxx = extract_power_spectral_feature(eeg_data, 250)
        # psd.append(pxx)
        #print(content.shape)  # 或者对内容进行其他处理






