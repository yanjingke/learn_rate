import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.layers import Flatten, Conv2D, Dropout, Input, Dense, MaxPooling2D
from keras.models import Model


def exponent(global_epoch,
             decay_steps,
             learning_rate_base,
             min_learn_rate=0,


             ):

    if global_epoch> 0:
            global_step = min(global_epoch, decay_steps)

            # linear_decay =(decay_steps-global_step)/decay_steps
            cosine_decay = 0.5 * (1 +np.cos(np.pi * global_step / decay_steps))

            decayed = (1 - min_learn_rate) * cosine_decay +  min_learn_rate


            learning_rate = learning_rate_base* decayed

    else:
            learning_rate = learning_rate_base

    learning_rate = max(learning_rate, min_learn_rate)


    return learning_rate


class COSScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """

    def __init__(self,
                 learning_rate_base,
                 decay_rate,
                 decay_steps,
                 global_epoch_init=0,
                 min_learn_rate=0.001,
                 verbose=0,
                ):
        super(COSScheduler, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 全局初始化epoch
        self.global_epoch = global_epoch_init
        self.decay_rate = decay_rate
        # 是否打印显示
        self.verbose = verbose
        self.decay_steps = decay_steps
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

    # 加入进去方便画图，并且设置学习率
    def on_epoch_end(self, batch, logs=None):
        self.global_epoch = self.global_epoch + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    # 更新学习率
    def on_epoch_begin(self, batch, logs=None):
        lr = exponent(global_epoch=self.global_epoch,
                      decay_steps=self.decay_steps,
                      learning_rate_base=self.learning_rate_base,

        min_learn_rate=self.min_learn_rate,

                      )
        # self.learning_rate_base = lr
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_epoch + 1, lr))


# 载入Mnist手写数据集
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# -----------------------------#
#   创建模型
# -----------------------------#
inputs = Input([28, 28, 1])
x = Conv2D(32, kernel_size=5, padding='same', activation="relu")(inputs)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', )(x)
x = Conv2D(64, kernel_size=5, padding='same', activation="relu")(x)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', )(x)
x = Flatten()(x)
x = Dense(1024)(x)
x = Dense(256)(x)
out = Dense(10, activation='softmax')(x)
model = Model(inputs, out)

# 设定优化器，loss，计算准确率
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 设置训练参数
epochs = 10

init_epoch = 0
# 每一次训练使用多少个Batch
batch_size = 31
# 最大学习率
learning_rate_base =1e-3

sample_count = len(x_train)

# 学习率
exponent_lr =COSScheduler(learning_rate_base=learning_rate_base,
                                      global_epoch_init=init_epoch,
                                      decay_rate=0.9,
                                      min_learn_rate=1e-8, verbose=1,
                                      decay_steps=2,


                                      )

# 利用fit进行训练
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
          verbose=1, callbacks=[exponent_lr])

plt.plot(exponent_lr.learning_rates)
plt.xlabel('Step', fontsize=20)
plt.ylabel('lr', fontsize=20)
plt.axis([0, epochs, 0, learning_rate_base * 1.1])
plt.xticks(np.arange(0, epochs, 1))
plt.grid()
plt.title('lr decay with COS_decay_Circle', fontsize=20)
plt.show()
