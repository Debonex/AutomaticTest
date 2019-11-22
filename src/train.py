import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# x_test = np.reshape(x_test,(10000,28,28,1))
# np.save('test_data.npy',x_test)
# np.save('./data/test_labels.npy',y_test)

# alist = [x_test[19],x_test[2],x_test[1],x_test[13],x_test[6],x_test[8],x_test[4],x_test[9],x_test[18],x_test[0]]
# nl = np.array(alist)
# nl = np.reshape(nl,(10,28,28,1))
# np.save('base.npy',nl)
# print(y_test[0:20])

# Sequential序列化模型
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28, 1)),# Flatten层用于将数据转化为一维数据 input_shape输入的尺寸
#   tf.keras.layers.Dense(128, activation='softplus'),# 定义层，128个神经元，激活函数为relu
#   tf.keras.layers.Dropout(0.2),# Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，用于防止过拟合。
#   tf.keras.layers.Dense(10, activation='softmax')
# ])


# 针对fashion_MINIST的神经网络 https://www.jianshu.com/p/96653fe0c74f
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
#   tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#   tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128,activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10,activation='softmax')
# ])


# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# x_train = np.reshape(x_train,(60000,28,28,1))
# x_test = np.reshape(x_test,(10000,28,28,1))

# model.fit(x_test, y_test, epochs=5)

# model.evaluate(x_test,  y_test, verbose=2)

# # 用于存储模型
# model.save(r'conv2d_model2.ht')