"""
    定义了SWA的回调函数类，
    用于在训练深度学习模型时执行Stochastic Weight Averaging（SWA，随机权重平均）优化技术
    这段代码实现了一个SWA的回调函数类，通过在训练过程中的不同阶段更新模型的权重，
    将之前的权重做个平均，作为模型的权重
    https://blog.csdn.net/zhaohongfei_358/article/details/129949398
    并根据SWA算法的特性对批标准化层进行适当的处理，以实现模型参数的平均化和泛化性能（模型在处理未见过的、新样本时的表现能力）的改善。
"""


def create_swa_callback_class(K, callback, batch_normalization):

    """Injecting library dependencies
    三个参数：K，callback和batch_normalization。其中，K通常是Keras的后端引擎（如TensorFlow或Theano）；
    callback是回调函数，用于训练过程中的回调操作；
    batch_normalization是批标准化的类，用于指定深度学习模型中的批标准化层。"""

    class SWA(callback):
        """Stochastic Weight Averging.这个类继承自传入的callback，表示它是一个回调函数

        # Paper
            title: Averaging Weights Leads to Wider Optima and Better Generalization
            link: https://arxiv.org/abs/1803.05407

        # Arguments
            start_epoch:   integer, epoch when swa should start.    SWA开始的epoch，表示在哪个epoch开始执行SWA
            lr_schedule:   string, type of learning rate schedule.  学习率调度类型，可以是"manual"、"constant"或"cyclic"之一
            swa_lr:        float, learning rate for swa.   SWA的学习率，用于更新模型参数的学习率
            swa_lr2:       float, upper bound of cyclic learning rate.  SWA的第二个学习率，对于周期性学习率调度方式，表示学习率的上限。
            swa_freq:      integer, length of learning rate cycle.    SWA的频率，表示执行SWA更新的周期
            batch_size     integer, batch size (for batch norm with generator)  批大小，用于批标准化层的更新
            verbose:       integer, verbosity mode, 0 or 1.   详细程度，控制输出信息的详细程度
        """

        """
        在类的构造函数中，定义了多个参数，包括 SWA 开始的 epoch、学习率调度、
        SWA 学习率、频率等。
        """
        def __init__(
            self,
            start_epoch,
            lr_schedule="manual",
            swa_lr="auto",
            swa_lr2="auto",
            swa_freq=1,
            batch_size=None,
            verbose=0,
        ):

            #调用父类的构造函数方法
            super().__init__()
            self.start_epoch = start_epoch - 1
            self.lr_schedule = lr_schedule
            self.swa_lr = swa_lr

            # if no user determined upper bound, make one based off of the lower bound,判断是否指定了值，如果没有指定，则使用默认值
            self.swa_lr2 = swa_lr2 if swa_lr2 is not None else 10 * swa_lr
            self.swa_freq = swa_freq
            self.batch_size = batch_size
            self.verbose = verbose

            if start_epoch < 2:
                raise ValueError('"swa_start" attribute cannot be lower than 2.')

            schedules = ["manual", "constant", "cyclic"]

            if self.lr_schedule not in schedules:
                raise ValueError(
                    '"{}" is not a valid learning rate schedule'.format(
                        self.lr_schedule
                    )
                )

            if self.lr_schedule == "cyclic" and self.swa_freq < 2:
                raise ValueError(
                    '"swa_freq" must be higher than 1 for cyclic schedule.'
                )

            if self.swa_lr == "auto" and self.swa_lr2 != "auto":
                raise ValueError(
                    '"swa_lr2" cannot be manually set if "swa_lr" is automatic.'
                )

            if (
                self.lr_schedule == "cyclic"
                and self.swa_lr != "auto"
                and self.swa_lr2 != "auto"
                and self.swa_lr > self.swa_lr2
            ):
                raise ValueError('"swa_lr" must be lower than "swa_lr2".')

        """
           回调函数的生命周期方法在训练开始时进行初始化，设置SWA的初始学习率和其他必要参数
         """
        def on_train_begin(self, logs=None):
            # 记录学习率的历史记录
            self.lr_record = []
            # 总的训练epoch数
            self.epochs = self.params.get("epochs")

            if self.start_epoch >= self.epochs - 1:
                raise ValueError('"swa_start" attribute must be lower than "epochs".')

            self.init_lr = K.eval(self.model.optimizer.lr)

            # automatic swa_lr
            if self.swa_lr == "auto":
                self.swa_lr = 0.1 * self.init_lr

            if self.init_lr < self.swa_lr:
                raise ValueError('"swa_lr" must be lower than rate set in optimizer.')

            # automatic swa_lr2 between initial lr and swa_lr
            if self.lr_schedule == "cyclic" and self.swa_lr2 == "auto":
                self.swa_lr2 = self.swa_lr + (self.init_lr - self.swa_lr) * 0.25

            self._check_batch_norm()

            if self.has_batch_norm and self.batch_size is None:
                raise ValueError(
                    '"batch_size" needs to be set for models with batch normalization layers.'
                )

        """
          回调函数的生命周期方法在每个epoch开始时，根据SWA的参数调整学习率，并在需要的时候重置批归一化层。
            """
        def on_epoch_begin(self, epoch, logs=None):
            # 设置当前的epoch
            self.current_epoch = epoch
            # 调用_scheduler方法
            self._scheduler(epoch)

            # 如果学习率调度为"constant"，则更新学习率
            if self.lr_schedule == "constant":
                self._update_lr(epoch)
            # 如果处于SWA开始的epoch，则保存当前模型的权重并打印消息
            if self.is_swa_start_epoch:
                self.swa_weights = self.model.get_weights()

                if self.verbose > 0:
                    print(
                        "\nEpoch %05d: starting stochastic weight averaging"
                        % (epoch + 1)
                    )
            # 如果处于需要重新初始化batch normalization的epoch，则设置SWA权重并打印消息
            if self.is_batch_norm_epoch:
                self._set_swa_weights(epoch)

                if self.verbose > 0:
                    print(
                        "\nEpoch %05d: reinitializing batch normalization layers"
                        % (epoch + 1)
                    )
                # 重置batch normalization并打印消息
                self._reset_batch_norm()

                if self.verbose > 0:
                    print(
                        "\nEpoch %05d: running forward pass to adjust batch normalization"
                        % (epoch + 1)
                    )

        """
           回调函数的生命周期方法在每个batch开始时，根据需要更新学习率，并根据SWA的参数设置批归一化的动量。
         """
        def on_batch_begin(self, batch, logs=None):

            # update lr each batch for cyclic lr schedule
            # 对于循环学习率调度，每个批次更新学习率
            if self.lr_schedule == "cyclic":
                self._update_lr(self.current_epoch, batch)
            # 如果是批次归一化的周期
            if self.is_batch_norm_epoch:

                batch_size = self.batch_size
                momentum = batch_size / (batch * batch_size + batch_size)
                # 对于每层批次归一化，更新动量值
                for layer in self.batch_norm_layers:
                    layer.momentum = momentum

        """
           回调函数的生命周期方法在每个epoch结束时，如果是SWA的轮数，则进行权重的平均化
         """
        def on_epoch_end(self, epoch, logs=None):

            if self.is_swa_start_epoch:
                self.swa_start_epoch = epoch

            if self.is_swa_epoch and not self.is_batch_norm_epoch:
                self.swa_weights = self._average_weights(epoch)

        """
           回调函数的生命周期方法
         """
        def on_train_end(self, logs=None):

            if not self.has_batch_norm:
                self._set_swa_weights(self.epochs)
            else:
                self._restore_batch_norm()

            for batch_lr in self.lr_record:
                self.model.history.history.setdefault("lr", []).append(batch_lr)

        """
           根据当前epoch确定是否执行SWA更新，是否达到SWA开始的epoch以及是否是批标准化的epoch。
         """
        def _scheduler(self, epoch):

            swa_epoch = epoch - self.start_epoch

            self.is_swa_epoch = (
                epoch >= self.start_epoch and swa_epoch % self.swa_freq == 0
            )
            self.is_swa_start_epoch = epoch == self.start_epoch
            self.is_batch_norm_epoch = epoch == self.epochs - 1 and self.has_batch_norm

        """
         计算SWA过程中的平均权重。
         """
        def _average_weights(self, epoch):

            return [
                (swa_w * ((epoch - self.start_epoch) / self.swa_freq) + w)
                / ((epoch - self.start_epoch) / self.swa_freq + 1)
                for swa_w, w in zip(self.swa_weights, self.model.get_weights())
            ]

        """
        更新学习率，根据不同的学习率调度策略进行更新
        """
        def _update_lr(self, epoch, batch=None):

            if self.is_batch_norm_epoch:
                lr = 0
                K.set_value(self.model.optimizer.lr, lr)
            elif self.lr_schedule == "constant":
                lr = self._constant_schedule(epoch)
                K.set_value(self.model.optimizer.lr, lr)
            elif self.lr_schedule == "cyclic":
                lr = self._cyclic_schedule(epoch, batch)
                K.set_value(self.model.optimizer.lr, lr)
            self.lr_record.append(lr)

        """
        _constant_schedule和_cyclic_schedule方法分别实现了
        常数学习率和周期学习率的更新策略。
        """
        def _constant_schedule(self, epoch):
            t = epoch / self.start_epoch
            lr_ratio = self.swa_lr / self.init_lr
            if t <= 0.5:
                factor = 1.0
            elif t <= 0.9:
                factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
            else:
                factor = lr_ratio
            return self.init_lr * factor

        def _cyclic_schedule(self, epoch, batch):
            """Designed after Section 3.1 of Averaging Weights Leads to
            Wider Optima and Better Generalization(https://arxiv.org/abs/1803.05407)
            """
            # steps are mini-batches per epoch, equal to training_samples / batch_size
            steps = self.params.get("steps")

            # occasionally steps parameter will not be set. We then calculate it ourselves
            if steps is None:
                steps = self.params["samples"] // self.params["batch_size"]

            swa_epoch = (epoch - self.start_epoch) % self.swa_freq
            cycle_length = self.swa_freq * steps

            # batch 0 indexed, so need to add 1
            i = (swa_epoch * steps) + (batch + 1)
            if epoch >= self.start_epoch:

                t = (((i - 1) % cycle_length) + 1) / cycle_length
                return (1 - t) * self.swa_lr2 + t * self.swa_lr
            else:
                return self._constant_schedule(epoch)

        """
        将模型权重设置为SWA的平均权重
        """
        def _set_swa_weights(self, epoch):

            self.model.set_weights(self.swa_weights)

            if self.verbose > 0:
                print(
                    "\nEpoch %05d: final model weights set to stochastic weight average"
                    % (epoch + 1)
                )

        """
        用于检查、重置和恢复批标准化层的状态。
        """
        def _check_batch_norm(self):

            self.batch_norm_momentums = []
            self.batch_norm_layers = []
            self.has_batch_norm = False
            self.running_bn_epoch = False

            for layer in self.model.layers:
                if issubclass(layer.__class__, batch_normalization):
                    self.has_batch_norm = True
                    self.batch_norm_momentums.append(layer.momentum)
                    self.batch_norm_layers.append(layer)

            if self.verbose > 0 and self.has_batch_norm:
                print(
                    "Model uses batch normalization. SWA will require last epoch "
                    "to be a forward pass and will run with no learning rate"
                )

        def _reset_batch_norm(self):

            for layer in self.batch_norm_layers:

                # to get properly initialized moving mean and moving variance weights
                # we initialize a new batch norm layer from the config of the existing
                # layer, build that layer, retrieve its reinitialized moving mean and
                # moving var weights and then delete the layer
                bn_config = layer.get_config()
                new_batch_norm = batch_normalization(**bn_config)
                new_batch_norm.build(layer.input_shape)
                new_moving_mean, new_moving_var = new_batch_norm.get_weights()[-2:]
                # get rid of the new_batch_norm layer
                del new_batch_norm
                # get the trained gamma and beta from the current batch norm layer
                trained_weights = layer.get_weights()
                new_weights = []
                # get gamma if exists
                if bn_config["scale"]:
                    new_weights.append(trained_weights.pop(0))
                # get beta if exists
                if bn_config["center"]:
                    new_weights.append(trained_weights.pop(0))
                new_weights += [new_moving_mean, new_moving_var]
                # set weights to trained gamma and beta, reinitialized mean and variance
                layer.set_weights(new_weights)

        def _restore_batch_norm(self):

            for layer, momentum in zip(
                self.batch_norm_layers, self.batch_norm_momentums
            ):
                layer.momentum = momentum

    return SWA
