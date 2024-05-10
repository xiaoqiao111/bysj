from utils1 import data_generator
from tcn import compiled_tcn
from sklearn.metrics import confusion_matrix   # 导入混淆矩阵计算工具
from swa import SWA
import pandas as pd
import numpy as np
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.layers.normalization import BatchNormalization
from callback import create_swa_callback_class
from keras.utils import to_categorical


def run_task():
    (x_train, y_train), (x_test, y_test) = data_generator()

    model = compiled_tcn(return_sequences=False,
                         num_feat=1,
                         num_classes=2,
                         nb_filters=32,
                         kernel_size=4,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=1,
                         max_len=x_train[0:1].shape[1],
                         use_skip_connections=True)
    # 打印训练集数据形状，标签形状，测试集数据形状，标签形状
    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()    # 打印模型结构

    epochs = 100   # 迭代次数
    start_epoch = 75   # SWA开始的迭代轮次
    swa = SWA(start_epoch=start_epoch,
              lr_schedule='constant',
              swa_lr=0.001,
              verbose=1)

    # start_epoch =2
    # swa = SWA(start_epoch=start_epoch,
    #       lr_schedule='cyclic',
    #       swa_lr=0.001,
    #       swa_lr2=0.003,
    #       swa_freq=3,
    #       batch_size=32, # needed when using batch norm
    #       verbose=1)

    # 使用tcn模型进行训练
    model.fit(x_train, y_train.squeeze().argmax(axis=1), callbacks=[swa], batch_size=64, epochs=epochs,
              validation_data=(x_test, y_test.squeeze().argmax(axis=1)))
    # 对测试集标签进行处理
    # y_test = to_categorical(y_test.squeeze().argmax(axis=1))
    y_test = (y_test.squeeze().argmax(axis=1))
    # 对测试集进行评估
    pre = model.evaluate(x_test, y_test, batch_size=32)
    # 打印测试集损失和准确率
    print('test_loss:', pre[0], '- test_acc:', pre[1])


if __name__ == '__main__':
    # 定义了一个用于实现随机加权平均（Stochastic Weight Averaging，SWA）的 TensorFlow Keras SWA 对象。
    SWA = create_swa_callback_class(K, Callback, BatchNormalization)
    run_task()





