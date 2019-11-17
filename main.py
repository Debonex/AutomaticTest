## test for generate

import generate
import numpy as np
import tensorflow as tf


images = np.load('./test_data/test_data.npy')
shape = images.shape
attack_images = generate.generate(images,shape)
tfSSIM = tf.image.ssim(tf.convert_to_tensor(images),tf.convert_to_tensor(attack_images),1.0)
print("ssim:",tfSSIM)

