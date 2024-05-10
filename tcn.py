"""
定义了一个基于时间卷积网络（TCN）的模型结构，并提供了一个编译后的TCN模型
"""
# from tensorflow.python.keras.layers import add,Input,Conv1D,Activation,Flatten,Dense
import inspect
import numpy as np
import pandas as pd
from keras.optimizers import RMSprop
from tqdm import tqdm
import json
# from keras.models import Model
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.constraints import unit_norm
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from margin_softmax import *
from tensorflow.python.keras.callbacks import Callback
from typing import List
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers.core import Activation, SpatialDropout1D, Dense, Lambda
# from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.engine.input_layer import Input
# from tensorflow.python.keras.layers import BatchNormalization
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.layers.normalization.layer_normalization import LayerNormalization
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
"""
#from tensorflow.python.layers.normalization import BatchNormalization
#from tensorflow.python.keras.layers import LayerNormalization
"""


# 判断一个数是否是 2 的幂次方
def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)


# 调整 dilation rate扩张率列表 使其为 2 的幂次方
def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations

# ResidualBlock 类定义了 WaveNet TCN 的残差块。在初始化函数中，它接受一系列参数，例如扩张率、卷积滤波器数量、卷积核大小、填充方式、激活函数等
class ResidualBlock(Layer):
    """
       定义了WaveNet TCN的残差块
       """

    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 activation: str = LeakyReLU(alpha=0.05),
                 dropout_rate: float = 0,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 **kwargs):
        """为WaveNet TCN定义了残差块
        一个残差块由两个卷积层组成，第一个卷积层的扩张率为dilation_rate，第二个卷积层的扩张率为1。
        第一个卷积层的卷积核大小为kernel_size，第二个卷积层的卷积核大小为1。
        第一个卷积层的激活函数为activation，第二个卷积层的激活函数为None。
        第一个卷积层的卷积核初始化器为kernel_initializer，第二个卷积层的卷积核初始化器为None。
        第一个卷积层的输出通道数为nb_filters，第二个卷积层的输出通道数为None。
        第一个卷积层的padding方式为padding，第二个卷积层的padding方式为'valid'。
        第一个卷积层的dropout比例为dropout_rate，第二个卷积层的dropout比例为0。
        第一个卷积层的是否使用批量归一化为use_batch_norm，第二个卷积层的是否使用批量归一化为False。
        第一个卷积层的是否使用层归一化为use_layer_norm，第二个卷积层的是否使用层归一化为False。

        Args:
            x: 模型中的前一层
            training: 指示层应该处于训练模式还是推断模式的布尔值
            dilation_rate: 用于此残差块的扩张因子
            nb_filters: 该块中要使用的卷积滤波器的数量
            kernel_size: 卷积核的大小
            padding: 卷积层中使用的填充，'same'或'causal'。
            activation: o = Activation(x + F(x))中使用的最终激活函数
            dropout_rate: 0到1之间的浮点数，输入单元的删除比例
            kernel_initializer: 内核权重矩阵（Conv1D）的初始化器
            use_batch_norm: 是否在残差层中使用批量归一化
            use_layer_norm: 是否在残差层中使用层归一化
            kwargs: 用于配置Layer类的任何初始化参数
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []   # 初始化内部层列表
        self.layers_outputs = []    # 初始化内部层输出列表
        self.shape_match_conv = None  # 初始化形状匹配卷积层
        self.res_output_shape = None   # 初始化残差块的输出形状
        self.final_activation = None  # 初始化最终激活层

        super(ResidualBlock, self).__init__(**kwargs)

    def _add_and_activate_layer(self, layer):
        """
        用于构建层的辅助函数
        Args:
            layer: 将层添加到内部层列表，并根据ResidualBlocK的当前输出形状构建它。更新当前输出形状。
        """
        self.layers.append(layer)  # 添加层到内部层列表
        self.layers[-1].build(self.res_output_shape)  # 构建层并更新当前输出形状
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)  # 更新当前输出形状

    def build(self, input_shape):
        """
                构建残差块，包括卷积层、归一化层、激活层、dropout层。
                Args:
                    input_shape: 输入形状
                """
        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.layers = []   # 清空内部层列表
            self.res_output_shape = input_shape   # 将当前输出形状设为输入形状

            for k in range(2):    # 循环两次，构建两个卷积层
                name = 'conv1D_{}'.format(k)
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    self._add_and_activate_layer(Conv1D(filters=self.nb_filters,
                                                        kernel_size=self.kernel_size,
                                                        dilation_rate=self.dilation_rate,
                                                        padding=self.padding,
                                                        name=name,
                                                        kernel_initializer=self.kernel_initializer))

                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:   # 如果使用批量归一化，添加和激活BatchNormalization层
                        self._add_and_activate_layer(BatchNormalization())
                    elif self.use_layer_norm:    # 如果使用层归一化，添加和激活LayerNormalization层
                        self._add_and_activate_layer(LayerNormalization())

                self._add_and_activate_layer(Activation('relu'))   # 添加和激活ReLU激活层
                self._add_and_activate_layer(SpatialDropout1D(rate=self.dropout_rate))

            if self.nb_filters != input_shape[-1]:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'matching_conv1D'
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = Conv1D(filters=self.nb_filters,
                                                   kernel_size=1,
                                                   padding='same',
                                                   name=name,
                                                   kernel_initializer=self.kernel_initializer)

            else:
                name = 'matching_identity'
                self.shape_match_conv = Lambda(lambda x: x, name=name)

            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self.final_activation = Activation(self.activation)
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation)

            super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None):
        """
        返回：一个元组，其中第一个元素是残差模型张量，第二个是跳过连接张量。
        """
        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
            self.layers_outputs.append(x)
        x2 = self.shape_match_conv(inputs)
        self.layers_outputs.append(x2)
        res_x = layers.add([x2, x])
        self.layers_outputs.append(res_x)

        res_act_x = self.final_activation(res_x)
        self.layers_outputs.append(res_act_x)
        return [res_act_x, x]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]


class TCN(Layer):
    """创建一个TCN层。

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: 卷积层中要使用的滤波器的数量
            kernel_size: 每个卷积层中要使用的卷积核大小
            dilations: 扩张列表
            nb_stacks: 使用的残差块的堆叠数
            padding: 卷积层中使用的填充
            use_skip_connections: 是否从输入到每个残差块添加跳过连接
            return_sequences: 是否在输出序列中返回最后一个输出或整个序列
            activation: 此TCN中残差块o = Activation(x + F(x))中使用的激活函数
            dropout_rate: 0到1之间的浮点数，输入单元的删除比例
            kernel_initializer: 卷积层的内核权重矩阵的初始化程序
            use_batch_norm: 是否在残差层中使用批量归一化
            use_layer_norm: 是否在残差层中使用层归一化
            kwargs: 用于配置Layer类的任何其他参数

        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=False,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 **kwargs):

        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None  # in case return_sequence=False
        self.output_slice_index = None  # in case return_sequence=False
        self.padding_same_and_time_dim_unknown = False  # edge case if padding='same' and time_dim = None

        if isinstance(self.nb_filters, list):
            assert len(self.nb_filters) == len(self.dilations)

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        # initialize parent class
        super(TCN, self).__init__(**kwargs)

    @property
    def receptive_field(self):
        assert_msg = 'The receptive field formula works only with power of two dilations.'
        assert all([is_power_of_two(i) for i in self.dilations]), assert_msg
        return self.kernel_size * self.nb_stacks * self.dilations[-1]

    def build(self, input_shape):

        # member to hold current output shape of the layer for building purposes
        self.build_output_shape = input_shape

        # list to hold all the member ResidualBlocks
        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                res_block_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=res_block_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          activation=self.activation,
                                                          dropout_rate=self.dropout_rate,
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          kernel_initializer=self.kernel_initializer,
                                                          name='residual_block_{}'.format(len(self.residual_blocks))))
                # build newest residual block
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        self.output_slice_index = None
        if self.padding == 'same':
            time = self.build_output_shape.as_list()[1]
            if time is not None:  # if time dimension is defined. e.g. shape = (bs, 500, input_dim).
                self.output_slice_index = int(self.build_output_shape.as_list()[1] / 2)
            else:
                # It will known at call time. c.f. self.call.
                self.padding_same_and_time_dim_unknown = True

        else:
            self.output_slice_index = -1  # causal case.
        self.slicer_layer = Lambda(lambda tt: tt[:, self.output_slice_index, :])

    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            batch_size = self.build_output_shape[0]
            batch_size = batch_size.value if hasattr(batch_size, 'value') else batch_size
            nb_filters = self.build_output_shape[-1]
            return [batch_size, nb_filters]
        else:
            # Compatibility tensorflow 1.x
            return [v.value if hasattr(v, 'value') else v for v in self.build_output_shape]

    def call(self, inputs, training=None):
        x = inputs
        self.layers_outputs = [x]
        self.skip_connections = []
        for layer in self.residual_blocks:
            try:
                x, skip_out = layer(x, training=training)
            except TypeError:  # compatibility with tensorflow 1.x
                x, skip_out = layer(K.cast(x, 'float32'), training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)

        if self.use_skip_connections:
            x = layers.add(self.skip_connections)
            self.layers_outputs.append(x)

        if not self.return_sequences:
            # case: time dimension is unknown. e.g. (bs, None, input_dim).
            if self.padding_same_and_time_dim_unknown:
                self.output_slice_index = K.shape(self.layers_outputs[-1])[1] // 2
            x = self.slicer_layer(x)
            self.layers_outputs.append(x)
        return x

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(TCN, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config


def compiled_tcn(num_feat,  # type: int
                 num_classes,  # type: int
                 nb_filters,  # type: int
                 kernel_size,  # type: int
                 dilations,  # type: List[int]
                 nb_stacks,  # type: int
                 max_len,  # type: int
                 output_len=1,  # type: int
                 padding='causal',  # type: str
                 use_skip_connections=True,  # type: bool
                 return_sequences=False,
                 regression=False,  # type: bool
                 dropout_rate=0.2,  # type: float
                 name='tcn',  # type: str,
                 kernel_initializer='he_normal',  # type: str,
                 activation='relu',  # type:str,
                 opt='adam',
                 lr=0.002,
                 use_batch_norm=False,
                 use_layer_norm=False):
    # type: (...) -> Model
    """Creates a compiled TCN model for a given task (i.e. regression or classification).
    Classification uses a sparse categorical loss. Please input class ids and not one-hot encodings.

    Args:
        num_feat: 输入数据的特征数量
        num_classes: 最后一个稠密层的大小，我们要预测多少个类
        nb_filters: 在卷积层中要使用的滤波器的数量
        kernel_size: 每个卷积层中要使用的卷积核大小
        dilations: 扩张列表，例如：[1, 2, 4, 8, 16, 32, 64]
        nb_stacks: 使用的残差块堆叠数
        max_len: 最大序列长度，如果序列长度是动态的，请使用None
        padding: 在卷积层中要使用的填充
        use_skip_connections: 是否从输入到每个残差块添加跳过连接
        return_sequences: 是否在输出序列中返回最后一个输出或整个序列
        regression: 输出是连续的还是离散的
        dropout_rate: 0到1之间的浮点数，输入单元的删除比例
        activation: o = Activation(x + F(x))中使用的激活函数
        name: 模型的名称
        kernel_initializer: 卷积层的内核权重矩阵的初始化程序
        opt: 优化器名
        lr: 学习率
        use_batch_norm: 是否在残差层中使用批量归一化
        use_layer_norm: 是否在残差层中使用层归一化
    Returns:
        A compiled keras TCN.
    """

    dilations = adjust_dilations(dilations)

    input_layer = Input(shape=(max_len, num_feat))

    x = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
            use_skip_connections, dropout_rate, return_sequences,
            activation, kernel_initializer, use_batch_norm, use_layer_norm,
            name=name)(input_layer)

    # print('x.shape=', x.shape)

    def get_opt():
        if opt == 'adam':
            return Adam(learning_rate=lr, clipnorm=1.)
        elif opt == 'rmsprop':
            return RMSprop(learning_rate=lr, clipnorm=1.)
        else:
            raise Exception('Only Adam and RMSProp are available here')

    if not regression:
        # classification
        x = Lambda(lambda x: K.l2_normalize(x, 1))(x)
        pred =Dense(num_classes)(x)
        encoder = Model(input_layer, x) # 最终的目的是要得到一个编码器
        model = Model(input_layer, pred)

        # https://github.com/keras-team/keras/pull/11373
        # It's now in Keras@master but still not available with pip.
        # TODO remove later.
        def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())
        #STCN
#       model.compile(optim, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        model.compile(get_opt(), loss=sparse_amsoftmax_loss, metrics=['sparse_categorical_accuracy'])
    else:
        # regression
        x = Dense(output_len)(x)
        x = Activation('linear')(x)
        output_layer = x
        model = Model(input_layer, output_layer)
        model.compile(get_opt(), loss='mean_squared_error')
    print('model.x = {}'.format(input_layer.shape))
    print('model.y = {}'.format(pred.shape))
    return model


def tcn_full_summary(model: Model, expand_residual_blocks=True):
    layers = model._layers.copy()  # store existing layers
    model._layers.clear()  # clear layers

    for i in range(len(layers)):
        if isinstance(layers[i], TCN):
            for layer in layers[i]._layers:
                if not isinstance(layer, ResidualBlock):
                    if not hasattr(layer, '__iter__'):
                        model._layers.append(layer)
                else:
                    if expand_residual_blocks:
                        for lyr in layer._layers:
                            if not hasattr(lyr, '__iter__'):
                                model._layers.append(lyr)
                    else:
                        model._layers.append(layer)
        else:
            model._layers.append(layers[i])

    model.summary()  # print summary

    # restore original layers
    model._layers.clear()
    [model._layers.append(lyr) for lyr in layers]
