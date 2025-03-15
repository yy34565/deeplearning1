# 简单的练手fashion数据集
import numpy as np #numpy模块用于数组和矩阵的运算
import pandas as pd #pandas模块用于处理表格数据
import matplotlib.pyplot as plt #用于绘图
import tensorflow as tf # 用于训练模型
# import keras    #高级神经网络的API，辅助tensorflow进行模型的训练
import warnings
warnings.filterwarnings('ignore')

fashion_minst=tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_minst.load_data()

class_names=['T-shit/top','Trouser','puillover','dress','coat','sandal','shirt','sneaker','bag','anklwe']
train_images,test_images = tf.cast(train_images/255.0,tf.float32),tf.cast(test_images/255.0,tf.float32)     #归一化
train_labels,test_labels = tf.cast(train_labels,tf.int16),tf.cast(test_labels,tf.int16)


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i],cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels])
# plt.show()

model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=10)

# 评估
test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)
print('\nTest accuracy:',test_acc)
print('\n',model.summary()) #简单输出模型的概要

predictions=model.predict(test_images)
print(predictions.shape)
print(predictions[0])#所属于10个类别的概率值
print(np.argmax(predictions[0]))


# 保存模型
model.save('fashion_modle.h5')
print('\n',model.summary()) #简单输出模型的概要

# 保存网络架构
# config=model.to_json()
#
# with open('config.json','w') as json:
#     json.write(config)
#
# # model=tf.keras.models.model_from_json(json_config)
#
# # 保存权重参数
# weights=model.get_weights()
# print(weights)
#
# model.save_weights('weights.h5') #保存权重参数
# model.load_weights('weights.h5')#加载权重参数






