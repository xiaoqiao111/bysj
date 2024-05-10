import numpy as np
from keras.utils import to_categorical
import scipy.io
import os
import scipy.signal
from keras.utils import np_utils
# 本句引用改成了keras.src.utils
from keras.models import Sequential
from keras import layers
import pandas as pd
import numpy as np
from keras import layers
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.feature_selection import SelectKBest,chi2
import pickle

def data_generator():


    data = scipy.io.loadmat("E:\\bysj_mk\psdfeature.mat")       #(3000,252)
    data = data['features']
    print(data.shape)
    # 加载标签
    label = scipy.io.loadmat("E:\\bysj_mk\psdfeature.mat")     #(3000,1)
    label = label['label']
    print(label.shape)

    # z-score标准化
    data = (data - data.mean(axis = 0))/(data.std(axis = 0))
    # 将data和label按照相同的随机方式8:2划分为x_train,x_test,y_train,和y_test。
    x_train, x_test, y_train, y_test = train_test_split(data,label, test_size=0.2, random_state=42)
    """print(x_train.shape)    # (2400,252)
    print(x_test.shape)     # (600,252)
    print(y_train.shape)    # (2400,1)
    print(y_test.shape)     # (600,1)"""
    num_classes = 2
    # to_categorical(y, num_classes)函数将输入的整数数组y（例如类标签）转换为独热编码的形式。
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    # 将数组y_train和y_test沿着第二个轴（axis=2）进行扩展，使它们的维度由(batch_size, num_classes)变为(batch_size, num_classes, 1)。
    y_train = np.expand_dims(y_train, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    print(data_generator())

 



