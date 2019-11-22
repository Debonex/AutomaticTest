import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# tensorflow2.0 教程中针对数字分类的模型参数 acc=88
# modelPath1 = r'./testmodels//primary_model.ht'

# https://www.jianshu.com/p/96653fe0c74f
modelPath1 = r'./models/model.ht'

# 
modelPath2 = r'./models/conv2d_model2.ht'

## 检验测试用
# descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
#                     for eps in epsilons]
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# 参数说明：
# images: 传入的测试样本数据(fashion-mnist数据集的shape的子集)，类型为numpy.ndarray。
# shape: images对应的shape，类型为tuple，例如image为1000张mnist的图片数据 (1000,28,28,1) 默认shape为(n, 28, 28 , 1), n为图片数据的数量
# return：返回基于images生成的对抗样本集合generate_images，二者shape一致，且一一对应（原始样本与对抗样本一一对应）
def generate(images,shape):
    
    startTime = time.clock()

    # 加载模型
    trained_model = tf.keras.models.load_model(modelPath1)

    # 模型对images预测的confidence ndarray,shape=(n,10)
    predicts = trained_model.predict(images)

    # 自训练模型对images的分类结果 ndarray,shape=(10000,)
    predict_types = get_type_list(predicts)

    adv_images = np.zeros(shape)
    succ_num = 0
    fail_num = 0
    for i in range(shape[0]):
        image = tf.convert_to_tensor(images[i])
        image = tf.reshape(image,(1,28,28,1))
        label = tf.one_hot(predict_types[i],predicts[i].shape[-1])

        # perturbations = create_adversarial_pattern(image,label,trained_model)
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = trained_model(image)
            loss = loss_object(label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, image)
        # Get the sign of the gradients to create the perturbation
        perturbations = tf.sign(gradient)
       
        flag = False
        eps = 0
        while eps < 0.2:
            eps += 0.01
            adv_image = image + eps*perturbations[0]
            adv_image = tf.clip_by_value(adv_image,0,1)          
            adv_predict = trained_model.predict(adv_image)
            adv_type = get_type_list(adv_predict)[0]          
            adv_confidence = max(adv_predict[0])
            if adv_type != predict_types[i] and adv_confidence>0.25:
                adv_images[i] = adv_image
                flag = True
                succ_num += 1
                sys.stdout.write(' '*50+'\r')
                sys.stdout.flush()
                sys.stdout.write(str(i+1)+'/'+str(shape[0]) +' 张图片处理完成. ' +
                                 str(succ_num)+' 张图片生成对抗样本成功. '+str(fail_num)+' 张图片生成对抗样本失败.\r')
                sys.stdout.flush()
                break
        if not flag:
            adv_image = image + 10*perturbations[0]
            adv_image = tf.clip_by_value(adv_image,0,1)
            adv_images[i] = adv_image
            fail_num += 1
            sys.stdout.write(' '*50+'\r')
            sys.stdout.flush()
            sys.stdout.write(str(i+1)+'/'+str(shape[0]) +' 张图片处理完成. ' +
                                 str(succ_num)+' 张图片生成对抗样本成功. '+str(fail_num)+' 张图片生成对抗样本失败.\r')
            sys.stdout.flush()

    print()
    print('处理完成，对抗样本生成成功率：'+str(succ_num/shape[0]*100)+'%')
    endTime = time.clock()
    print('Running time: %s Seconds'%(endTime-startTime))
    # save attack_data
    # np.save('attack_data1.npy',adv_images)
    return adv_images


def create_adversarial_pattern(input_image, input_label, trained_model):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = trained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad

def get_type_list(predicts):
    predicts_size = predicts.shape[0]
    types = np.zeros(predicts_size,dtype = np.int)
    for n in range(predicts_size):
        max = 0
        maxi = 0
        for i in range(predicts.shape[1]):
            if predicts[n][i]>max:
                max = predicts[n][i]
                maxi = i
        types[n] = maxi
    return types