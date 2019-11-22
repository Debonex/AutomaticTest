import tensorflow as tf
import numpy as np
import sys
import time

modelPath = r'./models/model.ht'

def generate(images,shape):

    startTime = time.clock()

    base_images = np.load('./data/base.npy')

    trained_model = tf.keras.models.load_model(modelPath)
    predicts = trained_model.predict(images)
    predict_types = tf.argmax(predicts,1)

    # 存储最终的对抗样本
    result_images = np.zeros(shape)

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    for i in range(shape[0]):
        image = tf.convert_to_tensor(images[i])
        image = tf.reshape(image,(1,28,28,1))
        label = tf.one_hot(predict_types[i],predicts[i].shape[-1])
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = trained_model(image)
            loss = loss_object(label, prediction)       
            gradient = tape.gradient(loss, image)       
        if i==0:
            perturbations = tf.sign(gradient)
        else:
            perturbations = tf.concat([perturbations,tf.sign(gradient)],axis=0)
        sys.stdout.write(' '*11+'\r')
        sys.stdout.flush()
        sys.stdout.write(str(i+1)+'/'+str(shape[0]) +' 张图片的噪声生成成功... \r')
        sys.stdout.flush()
    print()

    todolist = np.zeros(shape[0])
    eps = 0
    succ_num = 0
    while eps < 0.2:
        eps += 0.01
        adv_images = images + eps*perturbations
        adv_images = tf.clip_by_value(adv_images,0,1)
        adv_predicts = trained_model.predict(adv_images)
        adv_types = tf.argmax(adv_predicts,1)
        adv_confidence = tf.reduce_max(adv_predicts,1)
        for i in range(shape[0]):
            if todolist[i] == 0 and adv_types[i]!=predict_types[i] and adv_confidence[i]>0.4:
                result_images[i] = adv_images[i]
                todolist[i] = 1
                succ_num+=1
                sys.stdout.write(' '*11+'\r')
                sys.stdout.flush()
                sys.stdout.write(str(succ_num)+'/'+str(shape[0]) +' 张图片生成对抗样本成功... \r')
                sys.stdout.flush()
    print()
    print('正在处理剩余图片...')
    for i in range(shape[0]):
        if todolist[i] == 0:
            image_type = predict_types[i]
            result_images[i] = base_images[10%(image_type+1)] 
            # result_images[i] = images[i]     
    print('处理完成，对抗样本生成成功率：'+str(succ_num/shape[0]*100)+'%')
    endTime = time.clock()
    print('Running time: %s Seconds'%(endTime-startTime))
    # 存储对抗样本
    # np.save('attack_data.npy',result_images)
    return result_images