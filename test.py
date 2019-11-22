## test for generate

import generate
import numpy as np
import tensorflow as tf

num = 200
modelPath = r'./models/model.ht'
trained_model = tf.keras.models.load_model(modelPath)
labels = np.load('./data/test_labels.npy')
images = np.load('./data/test_data.npy')
images = np.reshape(images,(10000,28,28,1))
shape = images.shape


images = images[0:num]

shape = images.shape

# attack_images = generateT.generate(images,shape)
attack_images = generate.generate(images,shape)
attack_images = tf.convert_to_tensor(attack_images)

attack_images = tf.reshape(attack_images,(num,28,28,1))
images = tf.reshape(images,(num,28,28,1))

predicts = trained_model.predict(attack_images)
predict_types = tf.argmax(predicts,1)
count = 0
for i in range(shape[0]):
    if predict_types[i]!=labels[i]:
        count += 1
print('攻击成功率：'+str(count/num*100)+'%')


tfSSIM = tf.image.ssim(images,attack_images,1.0)

ssimlist = tfSSIM.numpy()
meanssim = np.mean(ssimlist)
print("ssim:",meanssim)
print("score:",count*meanssim/num)

