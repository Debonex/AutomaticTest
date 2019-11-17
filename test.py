# 安装 TensorFlow

import tensorflow as tf
import numpy as np
import cv2

mnist = tf.keras.datasets.fashion_mnist

text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                           'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
for i in y_test[0:32]:
  print(text_labels[i])

imgs = np.hstack(x_test[0:32])
cv2.imshow("xtrain",imgs)
cv2.waitKey(0)

# Sequential序列化模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),# Flatten层用于将数据转化为一维数据 input_shape输入的尺寸
  tf.keras.layers.Dense(128, activation='softplus'),# 定义层，128个神经元，激活函数为relu
  tf.keras.layers.Dropout(0.2),# Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，用于防止过拟合。
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)